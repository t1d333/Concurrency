all: build


build: 
	@go build -o lab1 ./cmd/main.go 
	
run:
	@go run ./cmd/main.go -n=$(size) -k=$(chunks)

test:
	@go test -v ./...

bench:
	@cd ./internal/matrixops/ && go test -bench .
	

clean:
	@rm ./lab1
