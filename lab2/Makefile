all: run


build: 
	@go build -tags mpi ./cmd/main.go


run: build
	@mpirun -np $(p) ./main -n=$(n)


clear:
	@rm ./main
	
