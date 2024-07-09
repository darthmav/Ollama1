package server

import (
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

type ModelPath struct {
	ProtocolScheme string
	Registry       string
	Namespace      string
	Repository     string
	Tag            string
}

const (
	DefaultRegistry       = "ollama.com"
	DefaultNamespace      = "library"
	DefaultTag            = "latest"
	DefaultProtocolScheme = "https"
)

var (
	ErrInvalidImageFormat  = errors.New("invalid image format")
	ErrInvalidProtocol     = errors.New("invalid protocol scheme")
	ErrInsecureProtocol    = errors.New("insecure protocol http")
	ErrInvalidDigestFormat = errors.New("invalid digest format")
)

func ParseModelPath(name string) ModelPath {
	mp := ModelPath{
		ProtocolScheme: DefaultProtocolScheme,
		Registry:       DefaultRegistry,
		Namespace:      DefaultNamespace,
		Repository:     "",
		Tag:            DefaultTag,
	}

	before, after, found := strings.Cut(name, "://")
	if found {
		mp.ProtocolScheme = before
		name = after
	}

	parts := strings.Split(filepath.ToSlash(name), "/")
	switch len(parts) {
	case 3:
		mp.Registry = parts[0]
		mp.Namespace = parts[1]
		mp.Repository = parts[2]
	case 2:
		mp.Namespace = parts[0]
		mp.Repository = parts[1]
	case 1:
		mp.Repository = parts[0]
	}

	if repo, tag, found := strings.Cut(mp.Repository, ":"); found {
		mp.Repository = repo
		mp.Tag = tag
	}

	return mp
}

var errModelPathInvalid = errors.New("invalid model path")

func (mp ModelPath) Validate() error {
	if mp.Repository == "" {
		return fmt.Errorf("%w: model repository name is required", errModelPathInvalid)
	}

	if strings.Contains(mp.Tag, ":") {
		return fmt.Errorf("%w: ':' (colon) is not allowed in tag names", errModelPathInvalid)
	}

	return nil
}

func (mp ModelPath) GetNamespaceRepository() string {
	return fmt.Sprintf("%s/%s", mp.Namespace, mp.Repository)
}

func (mp ModelPath) GetFullTagname() string {
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

func (mp ModelPath) GetShortTagname() string {
	if mp.Registry == DefaultRegistry {
		if mp.Namespace == DefaultNamespace {
			return fmt.Sprintf("%s:%s", mp.Repository, mp.Tag)
		}
		return fmt.Sprintf("%s/%s:%s", mp.Namespace, mp.Repository, mp.Tag)
	}
	return fmt.Sprintf("%s/%s/%s:%s", mp.Registry, mp.Namespace, mp.Repository, mp.Tag)
}

// GetManifestPath returns the path to the manifest file for the given model path, it is up to the caller to create the directory if it does not exist.
func (mp ModelPath) GetManifestPath() (string, error) {
	dir := envconfig.ModelsDir

	return filepath.Join(dir, "manifests", mp.Registry, mp.Namespace, mp.Repository, mp.Tag), nil
}

func (mp ModelPath) BaseURL() *url.URL {
	return &url.URL{
		Scheme: mp.ProtocolScheme,
		Host:   mp.Registry,
	}
}

func GetManifestPath() (string, error) {
	dir := envconfig.ModelsDir

	path := filepath.Join(dir, "manifests")
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "", err
	}

	return path, nil
}

func GetBlobsPath(digest string) (string, error) {
	dir := envconfig.ModelsDir

	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(dir, "blobs", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", err
	}

	return path, nil
}

func migrateRegistryDomain() error {
	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	targetDomain := filepath.Join(manifests, DefaultRegistry)
	if _, err := os.Stat(targetDomain); errors.Is(err, fs.ErrNotExist) {
		// noop
	} else if err != nil {
		return err
	} else {
		// target directory already exists so skip migration
		return nil
	}

	sourceDomain := filepath.Join(manifests, "registry.ollama.ai")

	//nolint:errcheck
	defer PruneDirectory(sourceDomain)

	return filepath.Walk(sourceDomain, func(source string, info fs.FileInfo, err error) error {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		} else if err != nil {
			return err
		}

		if !info.IsDir() {
			slog.Info("migrating registry domain", "path", source)

			rel, err := filepath.Rel(sourceDomain, source)
			if err != nil {
				return err
			}

			target := filepath.Join(targetDomain, rel)
			if _, err := os.Stat(target); errors.Is(err, fs.ErrNotExist) {
				// noop
			} else if err != nil {
				return err
			} else {
				return nil
			}

			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}

			if err := os.Rename(source, target); err != nil {
				return err
			}
		}

		return nil
	})
}
