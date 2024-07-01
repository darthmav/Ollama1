package convert

import (
	"archive/zip"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

type tempzipfs struct {
	r *zip.Reader
	p string
}

func TempZipFS(r *zip.Reader, p string) fs.FS {
	return &tempzipfs{r: r, p: p}
}

func (z *tempzipfs) Open(name string) (fs.File, error) {
	r, err := z.r.Open(name)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	if fi, err := r.Stat(); err != nil {
		return nil, err
	} else if fi.Size() < 32<<20 {
		return r, nil
	}

	if !filepath.IsLocal(name) {
		return nil, zip.ErrInsecurePath
	}

	n := filepath.Join(z.p, name)
	if _, err := os.Stat(n); errors.Is(err, os.ErrNotExist) {
		w, err := os.Create(n)
		if err != nil {
			return nil, err
		}
		defer w.Close()

		if _, err := io.Copy(w, r); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}

	return os.Open(n)
}
