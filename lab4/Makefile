all: run

CMD_FILE="./cmd/main.go"



clean:
	@rm -rf logs


run: clean
	@mkdir logs
	@go run ${CMD_FILE}
