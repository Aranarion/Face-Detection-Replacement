#include <netdb.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>

#define COMMAND_LINE_ERROR 16
#define INPUT_FILE_ERROR 13
#define OUTPUT_FILE_ERROR 5
#define PORT_ERROR 19
#define DEFAULT_CAPACITY 1000
#define ONE_BYTE 8
#define TWO_BYTES 16
#define THREE_BYTES 24
#define PREFIX_ONE 0x23
#define PREFIX_TWO 0x10
#define PREFIX_THREE 0x72
#define PREFIX_FOUR 0x31
#define PREIMAGE_SIZE 9
#define MASK 0xFF
#define THREE 3
#define FOUR 4
#define FIVE 5
#define SIX 6
#define SEVEN 7
#define EIGHT 8
#define PREFIX 0x23107231
#define COMMUNICATION_ERROR 9
#define ERROR_MESSAGE 11

/* Stores command-line parameters for uqfaceclient
 *
 * portnum: string of the port number if it's provided
 * detectFileGiven: 0 if no file given for detection, else 1
 * detectFile: the file name of the detection file if given
 * replaceFileGiven: 0 if no file given to replace, else 1
 * outputFileNameGiven: 0 if no output file name provided, else 1
 * outputFileName, string of the output file name if provided
 */
typedef struct {
    char* portnum;
    int detectFileGiven;
    char* detectFile;
    int replaceFileGiven;
    char* replaceFile;
    int outputFileNameGiven;
    char* outputFileName;
} CmdLineParams;

/* Buffer of binary code with specified length and data
 *
 * data: the binary code of the buffer
 * length: the size of the buffer
 */
typedef struct {
    uint8_t* data;
    size_t length;
} Buffer;

/* Sends command line error message which specifies the valid structure of a
 * command line and exits appropriately with exit code 16. To be called when
 * command line is invalid
 *
 *
 */
void cmd_line_error(void)
{
    fprintf(stderr,
            "Usage: ./uqfaceclient portnum [--replacefile filename] "
            "[--outputfilename filename] [--detectfile filename]\n");
    exit(COMMAND_LINE_ERROR);
}

/* Parses command-line arguments into CmdLineParams struct or if the command
 * line is invalid exits appropriately
 *
 * argc: The number of commandline arguments
 * argv: The array of commandline argument strings
 *
 * Returns CmdLineParams: struct containing the parsed parameters and flags
 */
CmdLineParams parse_command_line(int argc, char* argv[])
{
    CmdLineParams params = {0};
    if (argc < 2) {
        cmd_line_error();
    }
    if (strcmp(argv[1], "") == 0) {
        cmd_line_error();
    } else {
        params.portnum = argv[1];
    }
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--replacefile") == 0
                && params.replaceFileGiven == 0 && i + 1 < argc
                && strcmp(argv[i + 1], "")) {
            params.replaceFileGiven = 1;
            params.replaceFile = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--outputfilename") == 0
                && params.outputFileNameGiven == 0 && i + 1 < argc
                && strcmp(argv[i + 1], "")) {
            params.outputFileNameGiven = 1;
            params.outputFileName = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--detectfile") == 0
                && params.detectFileGiven == 0 && i + 1 < argc
                && strcmp(argv[i + 1], "")) {
            params.detectFileGiven = 1;
            params.detectFile = argv[i + 1];
            i++;
        } else {
            cmd_line_error();
        }
    }
    return params;
}

/* Prints an error message that the given input file cannot be opened for
 * reading and exits with exit code 13
 *
 * fileName: string that is the pathfile and file name of the input file
 *
 */
void input_file_error(char* fileName)
{
    fprintf(stderr,
            "uqfaceclient: unable to open the input file \"%s\" for reading\n",
            fileName);
    exit(INPUT_FILE_ERROR);
}

/* Prints an error message that the given output file cannot be opened for
 * writing and exits with exit code 5
 *
 * fileName: string of the pathfile and filename of the output file
 *
 */
void output_file_error(char* fileName)
{
    fprintf(stderr,
            "uqfaceclient: unable to open the output file \"%s\" for writing\n",
            fileName);
    exit(OUTPUT_FILE_ERROR);
}

/* Validates the accessibility of detection and replacement input files for
 * reading and output files for writing if these are specified
 *
 * params: The parsed commandline parameters containing flags if files and
 * specified and the filenames
 */
void check_files(CmdLineParams params)
{
    if (params.detectFileGiven) {
        FILE* file = fopen(params.detectFile, "r");
        if (!file) {
            input_file_error(params.detectFile);
        }
        fclose(file);
    }
    if (params.replaceFileGiven) {
        FILE* file = fopen(params.replaceFile, "r");
        if (!file) {
            input_file_error(params.replaceFile);
        }
        fclose(file);
    }
    if (params.outputFileNameGiven) {
        FILE* file = fopen(params.outputFileName, "w");
        if (!file) {
            output_file_error(params.outputFileName);
        }
        fclose(file);
    }
}

/* Prints an error message and exits with error code 19 if the server cannot
 * connect to the specified port
 *
 * port: string of the port specified on the commandline
 *
 */
void port_error(char* port)
{
    fprintf(stderr,
            "uqfaceclient: cannot connect to the server on port \"%s\"\n",
            port);
    exit(PORT_ERROR);
}

/* Attempts to connect to the specified port from command line or ephemoral port
 *
 * port: string of the port specified on the commandline
 *
 * returns fd: file descriptor for the connected socket on success
 */
int check_port(char* port)
{
    struct addrinfo* ai = 0;
    struct addrinfo hints;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    int err;
    if ((err = getaddrinfo("localhost", port, &hints, &ai))) {
        port_error(port);
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(fd, ai->ai_addr, sizeof(struct sockaddr))) {
        freeaddrinfo(ai);
        port_error(port);
    }
    freeaddrinfo(ai);
    return fd;
}

/* Reads contents of a file stream into a size-adjustable buffer, to accomodate
 * all data, returns a struct containing the data and the length of the data
 * read
 *
 * stream: Pointer to a file stream to read from
 *
 * Returns Buffer: A struct containing the read buffer data and the length read
 */
Buffer create_image_buffer(FILE* stream)
{
    size_t capacity = DEFAULT_CAPACITY, size = 0;
    uint8_t* bytes = malloc(capacity * sizeof(uint8_t));
    while (1) {
        if (size == capacity) {
            capacity *= 2;
            bytes = realloc(bytes, capacity * sizeof(uint8_t));
        }
        size_t new = fread(bytes + size, 1, capacity - size, stream);
        size += new;
        if (new == 0) {
            break;
        }
    }
    Buffer buffer = {0};
    buffer.data = bytes;
    buffer.length = size;
    return buffer;
}

/* Creates a request binary buffer combining one or two images and commandline
 * parameters.
 *
 * imageOne: The first image buffer to be included in the request
 * imageTwo: If replacement this is the image to replace onto faces
 * params: the parsed commandline arguments containing file name
 *
 * Returns Buffer: the full request to send to the server
 */
Buffer create_request(Buffer imageOne, Buffer imageTwo, CmdLineParams params)
{
    Buffer request = {0};
    size_t size = PREIMAGE_SIZE + imageOne.length;
    if (params.replaceFileGiven) {
        size += (FOUR + imageTwo.length);
    }
    uint8_t* buffer = malloc(size * sizeof(uint8_t));
    buffer[0] = PREFIX_FOUR;
    buffer[1] = PREFIX_THREE;
    buffer[2] = PREFIX_TWO;
    buffer[THREE] = PREFIX_ONE;
    if (params.replaceFileGiven) {
        buffer[FOUR] = 1;
    } else {
        buffer[FOUR] = 0;
    }
    uint32_t length = imageOne.length;
    buffer[FIVE] = (length)&MASK;
    buffer[SIX] = (length >> ONE_BYTE) & MASK;
    buffer[SEVEN] = (length >> TWO_BYTES) & MASK;
    buffer[EIGHT] = (length >> THREE_BYTES) & MASK;
    memcpy(buffer + PREIMAGE_SIZE, imageOne.data, imageOne.length);
    if (params.replaceFileGiven) {
        length = imageTwo.length;
        int index = PREIMAGE_SIZE + imageOne.length;
        buffer[index++] = length & MASK;
        buffer[index++] = (length >> ONE_BYTE) & MASK;
        buffer[index++] = (length >> TWO_BYTES) & MASK;
        buffer[index++] = (length >> THREE_BYTES) & MASK;
        memcpy(buffer + index, imageTwo.data, imageTwo.length);
    }
    request.data = buffer;
    request.length = size;
    return request;
}

/* Prints error message and exits with exit code 9 if there is a communication
 * error with the server
 */
void communication_error()
{
    fprintf(stderr, "uqfaceclient: unexpected communication error\n");
    exit(COMMUNICATION_ERROR);
}

/* Continues to read bytes from a file descriptor into a buffer until the
 * specified size is read in or if there is premature EOF or error
 *
 * fd: The file descriptor to read data in from
 * buffer: Pointer to the buffer where to store the read bytes
 * size: the requested number of bytes to read in
 *
 * Returns: -1 if an error or EOF occurs before reading all bytes or returns the
 * total number of bytes read on success
 */
ssize_t read_bytes(int fd, void* buffer, size_t size)
{
    size_t curr = 0;
    while (curr < size) {
        ssize_t new = read(fd, (char*)buffer + curr, size - curr);
        if (new <= 0) {
            return -1;
        }
        curr += new;
    }
    return curr;
}

/* Recieves and processes a response from the server via the given file
 * descriptor. For operation 2 the image data is written to stdout or to an
 * output file if specified. For operation 3 the error message is printed to
 * stderr and the program exits with the ERROR_MESSAGE code, else it's a
 * communication error.
 *
 * fd: The file descriptor to read the response from
 * params: Commandline parameters indicating output file options
 */
void recieve_response(int fd, CmdLineParams params)
{
    uint32_t prefix, imageSize;
    uint8_t operation;
    if (sizeof(prefix) != read_bytes(fd, &prefix, sizeof(prefix))) {
        communication_error();
    }
    if (prefix != PREFIX) {
        communication_error();
    }
    if (sizeof(operation) != read_bytes(fd, &operation, sizeof(operation))) {
        communication_error();
    }
    if (sizeof(imageSize) != read_bytes(fd, &imageSize, sizeof(imageSize))) {
        communication_error();
    }
    uint8_t* image = malloc(imageSize);
    if (imageSize != read_bytes(fd, image, imageSize)) {
        free(image);
        communication_error();
    }
    if (operation == 2) {
        FILE* output = stdout;
        if (params.outputFileNameGiven) {
            int fdOut = open(params.outputFileName,
                    O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
            output = fdopen(fdOut, "wb");
        }
        fwrite(image, 1, imageSize, output);
        fflush(output);
        if (params.outputFileNameGiven) {
            fclose(output);
        }
        free(image);
    } else if (operation == THREE) {
        fprintf(stderr,
                "uqfaceclient: received the following error message: \"");
        fwrite(image, 1, imageSize, stderr);
        fprintf(stderr, "\"\n");
        free(image);
        exit(ERROR_MESSAGE);
    } else {
        free(image);
        communication_error();
    }
}

/* Sends data over a socket suppressing SIGPIPE signals. Will send exactly the
 * specified length of bytes. Shall continue sending until all bytes are
 * transmitted or an error occurs
 *
 * fd: The file descriptor of the socket to send data to
 * data: Pointer to the data buffer to send
 * length: number of bytes to send
 *
 * Returns ssize_t The total number of bytes send on success of -1 if an error
 * occurs
 */
ssize_t send_without_signals(int fd, const void* data, size_t length)
{
    size_t curr = 0;
    const uint8_t* start = data;
    while (curr < length) {
        ssize_t new = send(fd, start + curr, length - curr, MSG_NOSIGNAL);
        if (new <= 0) {
            return -1;
        }
        curr += new;
    }
    return curr;
}

/* Parses command-line arguments and determines input and output files and a
 * portnumber or determines an ephemoral port to use.
 * Opens and reads image data from the input files or stdin and then constructs
 * a binary request using input images Sends the request to a server running on
 * the given port and recieves and handles the server's response which is to
 * save the image to the output file or send it to stdout Also sends error codes
 * for invalid command line, file errors and communication errors
 *
 * argc: Number of command line arguments
 * argv: Array of commandline argument strings
 * Returns in: 0 on successful execution, otherwise exits with an error code
 */
int main(int argc, char* argv[])
{
    CmdLineParams params = parse_command_line(argc, argv);
    check_files(params);
    int fd = check_port(params.portnum);
    FILE* stream = stdin;
    if (params.detectFileGiven) {
        stream = fopen(params.detectFile, "r");
    }
    Buffer imageOne = create_image_buffer(stream);
    if (params.detectFileGiven) {
        fclose(stream);
    }
    Buffer imageTwo = {0};
    if (params.replaceFileGiven) {
        stream = fopen(params.replaceFile, "r");
        imageTwo = create_image_buffer(stream);
        fclose(stream);
    }
    Buffer request = create_request(imageOne, imageTwo, params);
    if (send_without_signals(fd, request.data, request.length) == -1) {
        free(imageOne.data);
        free(imageTwo.data);
        communication_error();
    }
    free(imageOne.data);
    free(imageTwo.data);
    recieve_response(fd, params);
    return 0;
}
