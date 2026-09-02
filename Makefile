.PHONY: test vet lint

test:
	go test -race ./...

vet:
	go vet ./...

lint:
	golangci-lint run ./...
