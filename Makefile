.DEFAULT_GOAL := push

build:
	 docker build -t ollama:musa -f Dockerfile.musa -o type=docker .

push: build
	docker tag ollama:musa sh-harbor.mthreads.com/haive/ollama:musa
	docker push sh-harbor.mthreads.com/haive/ollama:musa