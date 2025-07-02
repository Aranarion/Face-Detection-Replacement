#include <netdb.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <ctype.h>
#include <stdint.h>
#include <limits.h>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/objdetect/objdetect_c.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>

#define ERROR_OPERATION 3
#define COMMAND_LINE_ERROR 19
#define PORTNUM_INDEX 3
#define EXTRA_CMD_LINE_ARGS 4
#define PORTNUM_INCLUDED 4
#define CASCADE_CLASSIFIER 14
#define OPERATION_ERROR 5
#define MAX_CONNECTIONS 10000
#define BASE 10
#define PORT_ERROR 10
#define FOUR_BYTES 4
#define MAX_IMAGE_SIZE 1000
#define PERMISSIONS 0666
#define LINE_THICKNESS 4
#define LINE_TYPE 8
#define PREFIX 0x23107231U
#define REVOLUTION 360
#define IMAGE_FILE_ERROR 18

/* Contains parsed command line flags and arguments
 *
 * connectionLimit: maximum number of clients that can be connected
 * maxSize: maximum image size
 * portnum: the portnumber string if provided
 * portnumGiven: 1 if portnum is given on command line else 0
 */
typedef struct {
    int connectionLimit;
    uint32_t maxSize;
    char* portnum;
    int portnumGiven;
} CmdLineParams;

/* Contains server statistics
 *
 * currClients: current number of clients connected
 * prevClients: number of clients that have connected and disconnected
 * faceDetections: number of successful face detection requests
 * faceReplacements: number of successful face replacement requests
 * statsLock: mutex for the statistics struct
 */
typedef struct {
    uint32_t currClients;
    uint32_t prevClients;
    uint32_t faceDetections;
    uint32_t faceReplacements;
    uint32_t badRequests;
    pthread_mutex_t statsLock;
} Statistics;

/* Struct which contains all pointers to be used in each thread to treat each
 * client
 *
 * clientfd: file descriptor of the client
 * limitConnections: pointer to semaphore that limits the total number of
 * connected clients fileLock: pointer to mutex that locks and unlocks the
 * temporary file cascadeLock: pointer to mutex that locks and unlocks the face
 * and eye cascades faceCascade: pointer to classifier that detects faces
 * eyesCascade: pointer to classifier that detects eyes
 * maxSize: maximum image size
 * stats: pointer to stats struct with current server stats
 */
typedef struct {
    int clientfd;
    sem_t* limitConnections;
    pthread_mutex_t* fileLock;
    pthread_mutex_t* cascadeLock;
    CvHaarClassifierCascade* faceCascade;
    CvHaarClassifierCascade* eyesCascade;
    uint32_t maxSize;
    Statistics* stats;
} ClientInfo;

/* Checks whether given string represents a valid integer commandline argument
 *
 * number: Null terminated string representing a number
 *
 * Returns int: 1 if valid, 0 if invalid
 */
int valid_cmd_line_number(char* number)
{
    if (number == NULL) {
        return 0;
    }
    int length = (int)strlen(number);
    if (length == 0) {
        return 0;
    }
    int i = 0;
    if (number[0] == '+') {
        if (length == 1) {
            return 0;
        }
        i++;
    }
    while (i < length) {
        if (!isdigit(number[i])) {
            return 0;
        }
        i++;
    }
    return 1;
}

/* Prints correct command line format and exits with exit code 19
 */
void command_line_error()
{
    fprintf(stderr,
            "Usage: ./uqfacedetect connectionlimit maxsize [portnumber]\n");
    exit(COMMAND_LINE_ERROR);
}

/* Parses and validates the command line arguments
 * Exits if argument is invalid or too few or many arguments are provided
 *
 * argc: number of commandline arguments
 * argv: array of commandline arguments
 *
 * Returns CmdLineParams: populated struct with arguments and flags
 */
CmdLineParams parse_command_line(int argc, char* argv[])
{
    CmdLineParams params = {0};
    params.portnum = "0";
    if (argc < PORTNUM_INDEX || argc > EXTRA_CMD_LINE_ARGS) {
        command_line_error();
    }
    if (valid_cmd_line_number(argv[1])) {
        params.connectionLimit = atoi(argv[1]);
        if (params.connectionLimit > MAX_CONNECTIONS) {
            command_line_error();
        }
    } else {
        command_line_error();
    }
    if (valid_cmd_line_number(argv[2])) {
        errno = 0;
        unsigned long test = strtoul(argv[2], NULL, BASE);
        if (errno == ERANGE || test > UINT32_MAX) {
            command_line_error();
        }
        params.maxSize = (uint32_t)test;
    } else {
        command_line_error();
    }
    if (params.maxSize == 0) {
        params.maxSize = UINT32_MAX;
    }
    if (argc == PORTNUM_INCLUDED) {
        if (strcmp(argv[PORTNUM_INDEX], "") == 0) {
            command_line_error();
        }
        if (valid_cmd_line_number(argv[PORTNUM_INDEX])
                && atoi(argv[PORTNUM_INDEX]) == 0) {
            return params;
        }
        params.portnum = argv[PORTNUM_INDEX];
        params.portnumGiven = 1;
    }
    return params;
}

/* Checks if image file "/tmp/imagefile.jpg" can be opened for writing by
 * attempting to write into file, else prints error message and exits with exit
 * code 18
 */
void check_image_file()
{
    FILE* image = fopen("/tmp/imagefile.jpg", "w");
    if (image == NULL) {
        fprintf(stderr,
                "uqfacedetect: unable to open image file for writing\n");
        exit(IMAGE_FILE_ERROR);
    }
    fclose(image);
}

/* Loads face and eyes Haar cascade classifiers for detection.
 * If either fails to load, an error message is printed and the program exits
 * with exit code 14
 *
 * faceCascade: pointer to a CvHaarClassifierCascade pointer where the face
 * classifier will be stored eyesCascade: pointer to a CvHaarClassifierCascade
 * pointer where the eyes classifier will be stored
 */
void check_cascade_classifier(CvHaarClassifierCascade** faceCascade,
        CvHaarClassifierCascade** eyesCascade)
{
    *faceCascade = (CvHaarClassifierCascade*)cvLoad(
            "/local/courses/csse2310/resources/a4/"
            "haarcascade_frontalface_alt2.xml",
            NULL, NULL, NULL);
    if (*faceCascade == NULL) {
        fprintf(stderr, "uqfacedetect: unable to load a cascade classifier\n");
        exit(CASCADE_CLASSIFIER);
    }
    *eyesCascade = (CvHaarClassifierCascade*)cvLoad(
            "/local/courses/csse2310/resources/a4/"
            "haarcascade_eye_tree_eyeglasses.xml",
            NULL, NULL, NULL);
    if (*eyesCascade == NULL) {
        fprintf(stderr, "uqfacedetect: unable to load a cascade classifier\n");
        exit(CASCADE_CLASSIFIER);
    }
}

/* Prints an error message and exits the program due to failure in binding to a
 * port
 */
void port_error(char* port)
{
    fprintf(stderr, "uqfacedetect: unable to listen on given port \"%s\"\n",
            port);
    exit(PORT_ERROR);
}

/* Creates and returns a listening socket bound to the specified port
 * If any step fails, an error message is printed along with the program exiting
 *
 * port: The string of portnumber on which server should listen
 *
 * Returns int: File descriptor for the listening socket if successful
 *
 */
int check_port(char* port)
{
    struct addrinfo* ai = 0;
    struct addrinfo hints;
    int listenfd, err, optval = 1;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    if ((err = getaddrinfo(NULL, port, &hints, &ai))) {
        freeaddrinfo(ai);
        port_error(port);
    }
    if ((listenfd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol))
            < 0) {
        freeaddrinfo(ai);
        port_error(port);
    }
    if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval))
            < 0) {
        close(listenfd);
        freeaddrinfo(ai);
        port_error(port);
    }
    if (bind(listenfd, ai->ai_addr, ai->ai_addrlen) < 0) {
        close(listenfd);
        freeaddrinfo(ai);
        port_error(port);
    }
    freeaddrinfo(ai);
    if (listen(listenfd, BASE) < 0) {
        close(listenfd);
        port_error(port);
    }
    return listenfd;
}

/* Prints the port number on which socket is listening to stderr
 *
 * listenfd: File descriptor of the listening socket
 */
void print_port(int listenfd)
{
    struct sockaddr_in ad;
    memset(&ad, 0, sizeof(struct sockaddr_in));
    socklen_t len = sizeof(struct sockaddr_in);
    getsockname(listenfd, (struct sockaddr*)&ad, &len);
    fprintf(stderr, "%d\n", ntohs(ad.sin_port));
    fflush(stderr);
}

/* Reads a specified number of bytes from a file descriptor into a buffer.
 * Continues until all bytes are read or error / EOF occurs
 *
 * fd: The file descriptor to read from
 * input: Pointer to the buffer where read bytes will be stored
 * totalSize: total number of bytes to read
 *
 * Returns ssize_t: Number of bytes read on success, else -1
 */
ssize_t read_bytes(int fd, void* input, size_t totalSize)
{
    size_t curr = 0;
    while (curr < totalSize) {
        ssize_t new = read(fd, (char*)input + curr, totalSize - curr);
        if (new <= 0) {
            return -1;
        }
        curr += new;
    }
    return curr;
}

/* Sends a specified number of bytes from a buffer into a file descriptor.
 * Continues until all bytes are sent or error / EOF occurs
 * Prevents SIGPIPE signals
 *
 * fd: The file descriptor to send to
 * input: Pointer to the buffer containing data to send
 * totalSize: total number of bytes to send
 *
 * Returns ssize_t: Number of bytes send on success, else -1
 */
ssize_t send_bytes(int fd, void* input, size_t totalSize)
{
    size_t curr = 0;
    while (curr < totalSize) {
        ssize_t new
                = send(fd, (char*)input + curr, totalSize - curr, MSG_NOSIGNAL);
        if (new <= 0) {
            return -1;
        }
        curr += new;
    }
    return curr;
}

/* Writes a specified number of bytes from a buffer into a file descriptor.
 * Continues until all bytes are written or error / EOF occurs
 *
 * fd: The file descriptor to write to
 * input: Pointer to the buffer containing data to write
 * totalSize: total number of bytes to write
 *
 * Returns ssize_t: Number of bytes written on success, else -1
 */
ssize_t write_bytes(int fd, void* input, size_t totalSize)
{
    size_t curr = 0;
    while (curr < totalSize) {
        ssize_t new = write(fd, (char*)input + curr, totalSize - curr);
        if (new <= 0) {
            return -1;
        }
        curr += new;
    }
    return curr;
}

/* Sends the formatted output message to a client socket consisting of prefix,
 * operation code, and 1-2 lengths of image and image
 *
 * clientfd: File descriptor of client socket to send data to
 * operation: Operation code byte to send
 * output: Pointer to the output data buffer to send
 * size: Size in bytes to send
 */
void send_output(
        int clientfd, uint8_t operation, unsigned char* output, uint32_t size)
{
    uint32_t prefix = htole32(PREFIX);
    uint32_t length = htole32(size);
    send_bytes(clientfd, &prefix, FOUR_BYTES);
    send_bytes(clientfd, &operation, 1);
    send_bytes(clientfd, &length, FOUR_BYTES);
    send_bytes(clientfd, output, size);
}

/* Sends the contents of predefined file to client socket when incorrect prefix
 * is recieved
 *
 * clientfd: file descriptor if client socket to send data to
 */
void send_prefix_file(int clientfd)
{
    int filefd = open(
            "/local/courses/csse2310/resources/a4/responsefile", O_RDONLY);
    struct stat stat;
    fstat(filefd, &stat);
    uint32_t fileSize = (uint32_t)stat.st_size;
    uint8_t* output = malloc(fileSize);
    read_bytes(filefd, output, fileSize);
    send_bytes(clientfd, output, fileSize);
    free(output);
    close(filefd);
}

/* Creates an OpenCV IplImage frame from image data by first writing image data
 * to a temporary file
 *
 * image: Pointer to binary representing image data buffer
 * size: number of bytes of image
 * fileLock: pointer to mutex for temporary file access
 *
 * Returns pointer to IplImage frame on success and NULL on failure
 */
IplImage* create_frame(
        unsigned char* image, uint32_t size, pthread_mutex_t* fileLock)
{
    pthread_mutex_lock(fileLock);
    int filefd = open(
            "/tmp/imagefile.jpg", O_RDWR | O_CREAT | O_TRUNC, PERMISSIONS);
    if (size != write_bytes(filefd, image, size)) {
        pthread_mutex_unlock(fileLock);
        close(filefd);
        return NULL;
    }
    close(filefd);
    IplImage* frame = cvLoadImage("/tmp/imagefile.jpg", CV_LOAD_IMAGE_COLOR);
    if (!frame) {
        pthread_mutex_unlock(fileLock);
        return NULL;
    }
    pthread_mutex_unlock(fileLock);
    return frame;
}

/* Converts a coloured IplImage frame to a greyscale, histogram-equalised image
 *
 * frame: Pointer to input coloured IplImage
 *
 * Return: Pointer to newly created grayscale, equalised IplImage
 */
IplImage* grey_image(IplImage* frame)
{
    IplImage* frameGrey = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
    cvCvtColor(frame, frameGrey, CV_BGR2GRAY);
    cvEqualizeHist(frameGrey, frameGrey);
    return frameGrey;
}

/* Detects faces in greyscale image using the classifier
 *
 * grey: Pointer to the input greyscale IplImage
 * cascade: Pointer to Haar cascade classifier for face detection
 * casecadeLock: Pointer to the mutex lock protecting the cascade classifier
 * storage: Pointer to the CvMemStorage pointer used for memory allocation
 * during detection
 *
 * Return: Pointer to a CvSeq structure containing detected faces
 */
CvSeq* find_faces(IplImage* grey, CvHaarClassifierCascade* cascade,
        pthread_mutex_t* cascadeLock, CvMemStorage** storage)
{
    const float haarScaleFactor = 1.1;
    pthread_mutex_lock(cascadeLock);
    if (*storage == NULL) {
        return NULL;
    }
    CvSeq* faces = cvHaarDetectObjects(grey, cascade, *storage, haarScaleFactor,
            LINE_THICKNESS, 0, cvSize(0, 0),
            cvSize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE));
    pthread_mutex_unlock(cascadeLock);
    return faces;
}

/* Draws detected faces and eyes on the given image frame
 *
 * frame: The color IplImage on which to draw face and eye annotations
 * frameGrey: grayscale version of the frame used for eye detection
 * faces: CvSeq of detected face rectangles
 * eyesCascade: Haar cascade classifier for eye detection
 * cascadeLock: Mutex protecting the cascade classifier during detection
 */
void draw_faces(IplImage* frame, IplImage* frameGrey, CvSeq* faces,
        CvHaarClassifierCascade* eyesCascade, pthread_mutex_t* cascadeLock)
{
    const float haarScaleFactor = 1.1;
    for (int i = 0; i < faces->total; i++) {
        CvRect* face = (CvRect*)cvGetSeqElem(faces, i);
        CvPoint center
                = {face->x + face->width / 2, face->y + face->height / 2};
        const CvScalar magenta = cvScalar(255, 0, 255, 0);
        const CvScalar blue = cvScalar(255, 0, 0, 0);
        cvEllipse(frame, center, cvSize(face->width / 2, face->height / 2), 0,
                0, REVOLUTION, magenta, LINE_THICKNESS, LINE_TYPE, 0);
        IplImage* faceROI
                = cvCreateImage(cvGetSize(frameGrey), IPL_DEPTH_8U, 1);
        cvCopy(frameGrey, faceROI, NULL);
        cvSetImageROI(faceROI, *face);
        pthread_mutex_lock(cascadeLock);
        CvMemStorage* eyeStorage = 0;
        eyeStorage = cvCreateMemStorage(0);
        cvClearMemStorage(eyeStorage);
        CvSeq* eyes = cvHaarDetectObjects(faceROI, eyesCascade, eyeStorage,
                haarScaleFactor, LINE_THICKNESS, 0, cvSize(0, 0),
                cvSize(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE));
        pthread_mutex_unlock(cascadeLock);
        if (eyes->total == 2) {
            for (int j = 0; j < eyes->total; j++) {
                CvRect* eye = (CvRect*)cvGetSeqElem(eyes, j);
                CvPoint eyeCenter = {face->x + eye->x + eye->width / 2,
                        face->y + eye->y + eye->height / 2};
                int radius = cvRound((eye->width / 2 + eye->height / 2) / 2);
                cvCircle(frame, eyeCenter, radius, blue, LINE_THICKNESS,
                        LINE_TYPE, 0);
            }
        }
        cvReleaseImage(&faceROI);
        cvReleaseMemStorage(&eyeStorage);
    }
}

/* Saves the processed image with detected faces to temporary file then reads
 * that file and send image data to client
 *
 * info: pointer to client information, including client file descriptor
 * fileLock: mutex to synchronize access to the image file
 * frame: IplImage containing the annotated image to be saved and sent
 * frameGrey: Grayscale version of the image
 * storage: memory storage used for face detection
 */
void output_face_detection(ClientInfo* info, pthread_mutex_t* fileLock,
        IplImage* frame, IplImage* frameGray, CvMemStorage* storage)
{
    pthread_mutex_lock(fileLock);
    FILE* file = fopen("/tmp/imagefile.jpg", "w");
    fclose(file);
    cvSaveImage("/tmp/imagefile.jpg", frame, 0);
    cvReleaseImage(&frame);
    cvReleaseImage(&frameGray);
    cvReleaseMemStorage(&storage);
    int imagefd = open("/tmp/imagefile.jpg", O_RDONLY);
    struct stat stat;
    fstat(imagefd, &stat);
    uint32_t outputSize = (uint32_t)stat.st_size;
    uint8_t* output = malloc(outputSize);
    read_bytes(imagefd, output, outputSize);
    close(imagefd);
    send_output(info->clientfd, 2, (unsigned char*)output, outputSize);
    free(output);
    pthread_mutex_unlock(fileLock);
}

/* Handles invalid message recieved from client by sending error response and
 * closing connection
 *
 * info: Pointer to ClientInfo struct containing client socket information
 */
void invalid_message(ClientInfo* info)
{
    send_output(info->clientfd, ERROR_OPERATION,
            (unsigned char*)"invalid message",
            (uint32_t)(sizeof("invalid message") - 1));
    shutdown(info->clientfd, SHUT_WR);
    // REF: Took inspiration from
    // https://stackoverflow.com/questions/4160347/close-vs-shutdown-socket
    close(info->clientfd);
}

/* Checks whether given image size is valid as in not 0 bytes and not exceeding
 * maximum
 *
 * imageSize: size of image to check
 * info: Pointer to ClientInfo containing client socket and max size limit
 *
 * Returns 1 if invalid and 0 if valid
 */
int check_image_size(uint32_t imageSize, ClientInfo* info)
{
    if (imageSize == 0) {
        send_output(info->clientfd, ERROR_OPERATION,
                (unsigned char*)"image is 0 bytes",
                (uint32_t)(sizeof("image is 0 bytes") - 1));
        shutdown(info->clientfd, SHUT_WR);
        close(info->clientfd);
        return 1;
    }
    if (info->maxSize != 0 && imageSize > info->maxSize) {
        send_output(info->clientfd, ERROR_OPERATION,
                (unsigned char*)"image too large",
                (uint32_t)(sizeof("image too large") - 1));
        shutdown(info->clientfd, SHUT_WR);
        close(info->clientfd);
        return 1;
    }
    return 0;
}

/* Reads replacement image from client saves it in temporary file and then loads
 * it as an image and returns this loaded image
 *
 * info: pointer to ClientInfo struct containing client file descriptor and
 * limits fileLock: pointer to a pthread mutex protecting file write/read
 * operations
 *
 * Returns pointer to loaded IplImage on success or NULL on failure
 */
IplImage* create_replacement(ClientInfo* info, pthread_mutex_t* fileLock)
{
    uint32_t imageTwoSizeLittleEndian;
    if (FOUR_BYTES
            != read_bytes(
                    info->clientfd, &imageTwoSizeLittleEndian, FOUR_BYTES)) {
        invalid_message(info);
        return NULL;
    }
    uint32_t imageTwoSize = le32toh(imageTwoSizeLittleEndian);
    if (check_image_size(imageTwoSize, info)) {
        return NULL;
    }
    unsigned char* image = malloc(imageTwoSize);
    if (read_bytes(info->clientfd, image, imageTwoSize) != imageTwoSize) {
        free(image);
        invalid_message(info);
        return NULL;
    }
    pthread_mutex_lock(fileLock);
    int filefd = open(
            "/tmp/imagefile.jpg", O_RDWR | O_CREAT | O_TRUNC, PERMISSIONS);
    if (imageTwoSize != write_bytes(filefd, image, imageTwoSize)) {
        pthread_mutex_unlock(fileLock);
        free(image);
        close(filefd);
        send_output(info->clientfd, ERROR_OPERATION,
                (unsigned char*)"invalid image",
                (uint32_t)(sizeof("invalid image") - 1));
        shutdown(info->clientfd, SHUT_WR);
        close(info->clientfd);
        return NULL;
    }
    close(filefd);
    free(image);
    IplImage* frame
            = cvLoadImage("/tmp/imagefile.jpg", CV_LOAD_IMAGE_UNCHANGED);
    if (!frame) {
        pthread_mutex_unlock(fileLock);
        send_output(info->clientfd, ERROR_OPERATION,
                (unsigned char*)"invalid image",
                (uint32_t)(sizeof("invalid image") - 1));
        shutdown(info->clientfd, SHUT_WR);
        close(info->clientfd);
        return NULL;
    }
    pthread_mutex_unlock(fileLock);
    return frame;
}

/* Replaces all detected faces in a frame with replacement image and sends
 * modified image back to client.
 *
 * info: Pointer to ClientInfo structure with socket and file lock
 * frame: Original frame with detected faces to be modified
 * replace: Image to overlay on all faces
 * frameGrey: Greyscale version of the frame
 * faces: Sequence of detected face regions
 * storage: Memory storage used during face detection
 */
void replace_face(ClientInfo* info, IplImage* frame, IplImage* replace,
        IplImage* frameGray, CvSeq* faces, CvMemStorage* storage)
{
    const int bgraChannels = 4;
    const int alphaIndex = 3;
    for (int i = 0; i < faces->total; i++) {
        CvRect* face = (CvRect*)cvGetSeqElem(faces, i);
        IplImage* resized = cvCreateImage(cvSize(face->width, face->height),
                IPL_DEPTH_8U, replace->nChannels);
        cvResize(replace, resized, CV_INTER_AREA);
        char* frameData = frame->imageData;
        char* faceData = resized->imageData;
        for (int y = 0; y < face->height; y++) {
            for (int x = 0; x < face->width; x++) {
                int faceIndex
                        = (resized->widthStep * y) + (x * resized->nChannels);
                if ((resized->nChannels == bgraChannels)
                        && (faceData[faceIndex + alphaIndex] == 0)) {
                    continue;
                }
                int frameIndex = (frame->widthStep * (face->y + y))
                        + ((face->x + x) * frame->nChannels);
                frameData[frameIndex + 0] = faceData[faceIndex + 0];
                frameData[frameIndex + 1] = faceData[faceIndex + 1];
                frameData[frameIndex + 2] = faceData[faceIndex + 2];
            }
        }
        cvReleaseImage(&resized);
    }
    output_face_detection(info, info->fileLock, frame, frameGray, storage);
    cvReleaseImage(&replace);
}

/* Handles a bad client request by updating stats and responding with default
 * file
 *
 * info: Pointer to ClientInfo structure containing socket and limits
 */
void bad_request(ClientInfo* info)
{
    pthread_mutex_lock(&(info->stats->statsLock));
    (info->stats->badRequests)++;
    pthread_mutex_unlock(&(info->stats->statsLock));
    send_prefix_file(info->clientfd);
    shutdown(info->clientfd, SHUT_WR);
    close(info->clientfd);
}

/* Handles request with invalid operation type by sending error response and
 * then closing socket.
 *
 * info: Pointer to ClientInfo structure containing client socket descriptor
 */
void wrong_operation(ClientInfo* info)
{
    send_output(info->clientfd, ERROR_OPERATION,
            (unsigned char*)"invalid operation type",
            (uint32_t)(sizeof("invalid operation type") - 1));
    shutdown(info->clientfd, SHUT_WR);
    close(info->clientfd);
}

/* Sends error response indicating invalid image was recieved
 *
 * info: Pointer to ClientInfo structure containing client socket descriptor
 */
void invalid_image(ClientInfo* info)
{
    send_output(info->clientfd, ERROR_OPERATION,
            (unsigned char*)"invalid image",
            (uint32_t)(sizeof("invalid image") - 1));
    shutdown(info->clientfd, SHUT_WR);
    close(info->clientfd);
}

/* Sends error response and closes client indicating no faces were found in the
 * image
 *
 * info: Pointer to ClientInfo structure containing client socket descriptor
 * storage: pointer toi memory storage used during detection
 *
 */
void no_faces(ClientInfo* info, CvMemStorage* storage, IplImage* grey,
        IplImage* frame)
{
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&grey);
    cvReleaseImage(&frame);
    send_output(info->clientfd, ERROR_OPERATION,
            (unsigned char*)"no faces detected in image",
            (uint32_t)(sizeof("no faces detected in image") - 1));
    shutdown(info->clientfd, SHUT_WR);
    close(info->clientfd);
}

/* Updates client statistics when a client disconnects.
 *
 * info A pointer to ClientInfo structure that contains a pointer to shared
 * statistics structure
 */
void update_client_stats(ClientInfo* info)
{

    pthread_mutex_lock(&(info->stats->statsLock));
    (info->stats->currClients)--;
    (info->stats->prevClients)++;
    pthread_mutex_unlock(&(info->stats->statsLock));
}

/* Executes face replacement on detected faces within a frame.
 *
 * info A pointer to a ClientInfo structure containing file and stats
 * information frame The original image frame in which faces are to be replaced
 * grey: greyscale version of frame used for detection
 * faces: Sequence of detected faces
 * storage: storage used by face detection
 *
 * Returns 0 on success, 1 on error
 */
int execute_replacement(ClientInfo* info, IplImage* frame, IplImage* grey,
        CvSeq* faces, CvMemStorage* storage)
{
    IplImage* replacement = create_replacement(info, info->fileLock);
    if (replacement == NULL) {
        return 1;
    }
    replace_face(info, frame, replacement, grey, faces, storage);
    pthread_mutex_lock(&(info->stats->statsLock));
    (info->stats->faceReplacements)++;
    pthread_mutex_unlock(&(info->stats->statsLock));
    return 0;
}

/* Validqates prefix of an incoming client message
 *
 * info A pointer to a ClientInfo structure containing the client file
 * descriptor
 *
 * Returns 0 if prefix is valid, else 1
 */
int check_prefix(ClientInfo* info)
{
    uint32_t prefix;
    if (FOUR_BYTES != read_bytes(info->clientfd, &prefix, FOUR_BYTES)) {
        invalid_message(info);
        return 1;
    }
    if (le32toh(prefix) != PREFIX) {
        bad_request(info);
        return 1;
    }
    return 0;
}

/* Reads and validates the operation code from the client
 *
 * info: A pointer to a ClientInfo structure containing the client file
 * descriptor.
 *
 * Return operation (0,1) if valid else returns 5
 */
uint8_t check_operation(ClientInfo* info)
{
    uint8_t operation;
    if (1 != read_bytes(info->clientfd, &operation, 1)) {
        invalid_message(info);
        return OPERATION_ERROR;
    }
    if (operation > 1) {
        wrong_operation(info);
        return OPERATION_ERROR;
    }
    return operation;
}

/* Reads and validates the size of the incoming image from the client
 *
 * Info: A pointer to a ClientInfo structure containing the client file
 * descriptor
 *
 * Returns validated image size in bytes or 0 if an error occurs
 */
uint32_t determine_image_size(ClientInfo* info)
{
    uint32_t imageOneSizeLittleEndian;
    if (FOUR_BYTES
            != read_bytes(
                    info->clientfd, &imageOneSizeLittleEndian, FOUR_BYTES)) {
        invalid_message(info);
        return 0;
    }
    uint32_t imageOneSize = le32toh(imageOneSizeLittleEndian);
    if (check_image_size(imageOneSize, info)) {
        return 0;
    }
    return imageOneSize;
}

/* Processes an image recieved from the client based on the requested operation.
 * Reconstructs image from raw data performs face detection or face detection.
 * If no faces are found or any processing step fails returns as error
 *
 * info: A pointer to a ClientInfo structure containing client-specific data and
 * locks image: A pointer to the raw image data received from the client
 * imageOneSize: Size of raw image data in b7ytes
 * operation: Operation to perform 0 for face detection, 1 for face replacement
 *
 * Returns 0 on success, 1 on failure
 */
int image_executor(ClientInfo* info, unsigned char* image,
        uint32_t imageOneSize, uint8_t operation)
{
    IplImage* frame = create_frame(image, imageOneSize, info->fileLock);
    if (frame == NULL) {
        invalid_image(info);
        return 1;
    }
    IplImage* grey = grey_image(frame);
    CvMemStorage* storage = 0;
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);
    CvSeq* faces
            = find_faces(grey, info->faceCascade, info->cascadeLock, &storage);
    if (faces == NULL || faces->total == 0) {
        no_faces(info, storage, grey, frame);
        return 1;
    }
    if (operation == 0) {
        draw_faces(frame, grey, faces, info->eyesCascade, info->cascadeLock);
        output_face_detection(info, info->fileLock, frame, grey, storage);
        pthread_mutex_lock(&(info->stats->statsLock));
        (info->stats->faceDetections)++;
        pthread_mutex_unlock(&(info->stats->statsLock));
    } else {
        if (execute_replacement(info, frame, grey, faces, storage)) {
            return 1;
        }
    }
    return 0;
}

/* Thread entry point for client's image processing request.
 * Processes incoming client data by validating prefix, check requested
 * operation type, determining size of incoming image and executes corresponding
 * image operation. Loop continues until error occurs or invalid input.
 *
 * arg: Void pointer to ClientInfo struct which contains all info
 *
 * Returns NULL
 */
void* task_executor(void* arg)
{
    ClientInfo* info = arg;
    while (1) {
        if (check_prefix(info)) {
            break;
        }
        uint8_t operation = check_operation(info);
        if (operation == OPERATION_ERROR) {
            break;
        }
        uint32_t imageOneSize = determine_image_size(info);
        if (0 == imageOneSize) {
            break;
        }
        unsigned char* image = malloc(imageOneSize);
        if (read_bytes(info->clientfd, image, imageOneSize) != imageOneSize) {
            free(image);
            invalid_message(info);
            break;
        }
        if (image_executor(info, image, imageOneSize, operation)) {
            free(image);
            break;
        }
        free(image);
    }
    if (info->limitConnections != NULL) {
        sem_post(info->limitConnections);
    }
    update_client_stats(info);
    free(info);
    return NULL;
}

/* Accepts new clien connections and spawns threads to handle them with
 * necessary information
 *
 * listenfd: socket file descriptor
 * limitConnections: optional pointer to semaphore limiting concurrent
 * connections fileLock: Pointer to mutex to control temporary file access
 * cascadeLock: Pointer to mutex to control face detection classifier access
 * faceCascade: Pointer to the Haar cascade classifier for detecting faces
 * eyesCascade: Pointer to the Haar cascade classifier for detecting eyes
 * maxSize: Maximum size for images
 * stats: Pointer to the shared Statistics structure used for tracking client
 * metrics
 */
void new_connection(int listenfd, sem_t* limitConnections,
        pthread_mutex_t* fileLock, pthread_mutex_t* cascadeLock,
        CvHaarClassifierCascade* faceCascade,
        CvHaarClassifierCascade* eyesCascade, uint32_t maxSize,
        Statistics* stats)
{
    while (1) {
        if (limitConnections) {
            sem_wait(limitConnections);
        }
        int clientfd = accept(listenfd, NULL, NULL);
        pthread_mutex_lock(&(stats->statsLock));
        (stats->currClients)++;
        pthread_mutex_unlock(&(stats->statsLock));
        ClientInfo* clientInfo = malloc(sizeof(ClientInfo));
        clientInfo->clientfd = clientfd;
        clientInfo->limitConnections = limitConnections;
        clientInfo->fileLock = fileLock;
        clientInfo->cascadeLock = cascadeLock;
        clientInfo->faceCascade = faceCascade;
        clientInfo->eyesCascade = eyesCascade;
        clientInfo->maxSize = maxSize;
        clientInfo->stats = stats;
        pthread_t tid;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_create(&tid, &attr, task_executor, clientInfo);
        pthread_attr_destroy(&attr);
    }
}

/* Thread that listens for SIGHUP signals and prints server statistics to stderr
 *
 * arg: void pointer to a Statistics structure containing stats to be printed
 *
 * returns NULL
 */
void* sighup_listener(void* arg)
{
    Statistics* stats = arg;
    sigset_t signalSet;
    int sig;
    sigemptyset(&signalSet);
    sigaddset(&signalSet, SIGHUP);
    // REF: Took inspiration from:
    // https://stackoverflow.com/questions/21552600/sigwait-and-signal-handler
    while (1) {
        sigwait(&signalSet, &sig);
        pthread_mutex_lock(&(stats->statsLock));
        fprintf(stderr, "Num clients connected: %u\n", stats->currClients);
        fprintf(stderr, "Clients completed: %u\n", stats->prevClients);
        fprintf(stderr, "Face detect requests: %u\n", stats->faceDetections);
        fprintf(stderr, "Face replace requests: %u\n", stats->faceReplacements);
        fprintf(stderr, "Malformed requests: %u\n", stats->badRequests);
        fflush(stderr);
        pthread_mutex_unlock(&(stats->statsLock));
    }
}

/* Sets up statistics tracking and a thread for this
 * Parses commandline parameters and validates files and classifiers
 * Sets up listening socket on specified port
 * Initialises connection limit via semaphore
 * Enters loop to accept new connections
 *
 * argc Count of command line arguments
 * argv Array of command-line argument strings
 *
 * Returns 0
 */
int main(int argc, char* argv[])
{
    Statistics stats = {0};
    pthread_mutex_init(&stats.statsLock, NULL);
    sigset_t signalSet;
    sigemptyset(&signalSet);
    sigaddset(&signalSet, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &signalSet, NULL);
    pthread_t thread;
    pthread_create(&thread, NULL, sighup_listener, &stats);
    // REF Took inspiration from:
    // https://stackoverflow.com/questions/21552600/sigwait-and-signal-handler
    CmdLineParams params = parse_command_line(argc, argv);
    check_image_file();
    CvHaarClassifierCascade *faceCascade, *eyeCascade;
    pthread_mutex_t cascadeLock, fileLock;
    pthread_mutex_init(&cascadeLock, NULL);
    pthread_mutex_init(&fileLock, NULL);
    check_cascade_classifier(&faceCascade, &eyeCascade);
    int listenfd = check_port(params.portnum);
    print_port(listenfd);
    sem_t* limitConnections = NULL;
    if (params.connectionLimit != 0) {
        limitConnections = malloc(sizeof(sem_t));
        sem_init(limitConnections, 0, params.connectionLimit);
    }
    new_connection(listenfd, limitConnections, &fileLock, &cascadeLock,
            faceCascade, eyeCascade, params.maxSize, &stats);
    cvReleaseHaarClassifierCascade(&faceCascade);
    cvReleaseHaarClassifierCascade(&eyeCascade);
    free(limitConnections);
    return 0;
}
