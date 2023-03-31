#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

struct Bucket {
    float value;
    struct Bucket* next;
    int count;
};

int bucketsort_parallel(float* floatArrayToSort, int size,
                         int bucketCount, int itemsPerProcessor);
void getDistributions(int *start, int *portion, int remainder, int rank);
void initialiseBuckets(struct Bucket* buckets, int bucketCount);
int fillBuckets(const float* floatArrayToSort, int size,
                 struct Bucket* buckets, int bucketCount);
void printBuckets(struct Bucket* buckets, int bucketCount);
void mergesort_parallel(float* floatArrayToSort, int size, MPI_Comm communicator);
void mergesort(float* array, int low, int high);
void merge(float* floatArrayToSort, int low, int mid, int high);
void makeGathervCall(void* recvbuf, int rank, int processorCount, int portion,
                     int remainder, int size, MPI_Comm communicator);


int bucketsort(float* floatArrayToSort, int size, int itemsPerProcessor) {
    return bucketsort_parallel(floatArrayToSort, size, 10, itemsPerProcessor);
}

int bucketsort_parallel(float* floatArrayToSort, int size,
                         int bucketCount, int itemsPerProcessor) {
    /*
     * floatArrayToSort: array to sort, self-descriptive
     * size: size of the array to sort
     * bucketCount: number of buckets
     * itemsPerProcessor: {bucketCount} additional processors allowed for every ..-
     * -.. extra {itemsPerProcessor} items in a bucket to allow parallel mergesort
     */
    if (bucketCount < 1) return 0;
    else if (bucketCount == 1) {
        mergesort(floatArrayToSort, 0, size-1);
    }
    int processorCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processorCount);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Ensure max 1 extra processor for each itemsPerProcessor in each bucket
//    WORK ON THE EQUATION BELOW. ALSO BUCKETCOUNT IS 1. IT SHOULD BE 2 OR 10 TO TEST.
    if (processorCount > (bucketCount +
            size/(bucketCount*itemsPerProcessor))) {
        if (!rank) {
            printf("Number of processors exceeds the allowed limit. There can be "
                   "one additional processor per bucket for every %d items (avg) "
                   "in each bucket.", itemsPerProcessor);
        }
        return 0;
    }

    struct Bucket* buckets = (struct Bucket*) malloc((size_t) bucketCount * sizeof(struct Bucket));
    if (buckets == NULL) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 0;
    }

    int portion, remainder, start, groupRank;
    groupRank = rank % bucketCount;
    portion = bucketCount / processorCount;
    remainder = bucketCount % processorCount;

    getDistributions(&start, &portion, remainder, groupRank);

    initialiseBuckets(buckets, bucketCount);
    // negative numbers in list or failure to malloc
    if (!fillBuckets(floatArrayToSort, size, buckets, bucketCount)) {
        free(buckets);
        return 0;
    }
    /* testing if buckets have been initialised correctly
     * if (!rank) {
     *    struct Bucket* b;
     *    float max[10] = {0}; // 10 = bucketCount
     *    for (int i = 0; i < bucketCount; i++) {
     *        b = &buckets[i];
     *        while (b != NULL) {
     *            if (b->value > max[i]) max[i] = b->value;
     *            b = b->next;
     *        }
     *    }
     *    for (int i = 1; i < 10; i++) {
     *        if (max[i] <= max[i-1]) {
     *            printf("i: %d, max[i]: %f, max[i-1]: %f\n");
     *        }
     *    }
     *    return 1;
     * }
     * return 0;
     */

    if (rank) { // Rank 0 keeps original array for to have space for final gather
        size_t newSize = portion * sizeof(struct Bucket);
        memcpy(buckets, buckets + start, newSize);
        buckets = (struct Bucket *) realloc(buckets, newSize);
        if (buckets == NULL) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 0;
        }
    }
    /* Create a new communicator for each bucket for parallel internal sort */
    MPI_Comm localCommunicator;
    MPI_Comm_split(MPI_COMM_WORLD, groupRank, rank, &localCommunicator);
    /* Gather items in each bucket and store in a separate array */
    int itemsInBucket;
    struct Bucket* currentBucket;
    struct Bucket* prevBucket;
    float* numbersInBuckets[portion];
    int sizes[portion]; // size of each bucket
//    Maybe free buckets in the loop below instead as soon as the data is copied? like in line 98 and somewhere around 110?
    for (int i = 0; i < portion; i++) {
        currentBucket = &buckets[i];
        itemsInBucket = currentBucket->count;
        sizes[i] = itemsInBucket;
        numbersInBuckets[i] = (float*) malloc(itemsInBucket * sizeof(float));
        if (!itemsInBucket) {
            continue;
        }
        for (int j = 0; j < itemsInBucket; j++) {
            numbersInBuckets[i][j] = currentBucket->value;
            prevBucket = currentBucket;
            currentBucket = currentBucket->next;
            if (j == 0) continue; // first bucket allocated on stack
            // Freeing buckets here saves the need to use another loop to do so.
            free(prevBucket);
        }
        mergesort_parallel(numbersInBuckets[i], itemsInBucket, localCommunicator);
    }
    free(buckets);

    /* Total number of items sorted */
    int totalItems = 0;
    // Auxillary processors only help with mergesort, thus don't run below
    if (rank < bucketCount) {
        for (int i = 0; i < portion; i++) {
            totalItems += sizes[i];
        }
    }
    float sortedArray[totalItems];
    /* Add all sorted numbers into the allocated array */
    if (rank < bucketCount) {
        int k = 0;
        for (int i = 0; i < portion; i++) {
            for (int j = 0; j < sizes[i]; j++) {
                sortedArray[k] = numbersInBuckets[i][j];
                k++;
            }
        }
    }
    for (int i = 0; i < portion; i++) {
        free(numbersInBuckets[i]);
    }

    /* Find recvcounts and displs for final Gatherv */
    if (!rank) {
        int recvcounts[processorCount], displs[processorCount];
        // Fill recvcounts
        MPI_Gather(&totalItems, 1, MPI_INT, recvcounts, 1, MPI_INT,
                   0, MPI_COMM_WORLD);
        // Work out displacements
        displs[0] = 0;
        for (int i = 1; i < processorCount; i++) {
            displs[i] = recvcounts[i - 1] + displs[i - 1];
        }
        // Final gatherv
        MPI_Gatherv(sortedArray, totalItems, MPI_FLOAT, floatArrayToSort,
                    recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Gather(&totalItems, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        /* Gather all sorted arrays into a unified array at rank 0 */
        MPI_Gatherv(sortedArray, totalItems, MPI_FLOAT, NULL,
                    NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    /* testing - printing sorted array */
//    if (!rank) {
//        for (int i = 0; i < size; i++) {
//            printf("%f\n", floatArrayToSort[i]);
//        }
//    }
    return rank == 0;
    /*
     * Create new communicators for each bucket.
     * This allows multiple processors to be run on each.
     * Hence, allowing mergesort to be done in parallel on each bucket.
     * The new communicators allow processor ranks to be local.
     * This allows for rank based parallelism.
     */
}

void getDistributions(int *start, int *portion, int remainder, int rank) {
    if (rank < remainder) { // Spread remainder evenly across processors
        (*portion)++; // Processors get a max 1 remainder bucket to deal with
        *start = (*portion)*rank; // All processors before have had remainder buckets
    }
    else {
        *start = ((*portion)+1)*remainder + (*portion)*(rank-remainder);
    }
}

void initialiseBuckets(struct Bucket* buckets, int bucketCount) {
    for (int i = 0; i < bucketCount; i++) {
        buckets[i].value = -1;
        buckets[i].next = NULL;
        buckets[i].count = 0;
    }
}

void freeBuckets(struct Bucket* buckets, int bucketCount) {
    struct Bucket *prevBucket, *currentBucket;
    for (int i = 0; i < bucketCount; i++) {
        currentBucket = &buckets[i];
        for (int j = 0; j < currentBucket->count; j++) {
            prevBucket = currentBucket;
            currentBucket = currentBucket->next;
            if (j == 0) continue; // Ignore first bucket allocated on stack
            free(prevBucket);
        }
    }
    free(buckets);
}

int fillBuckets(const float* floatArrayToSort, int size, struct Bucket* buckets, int bucketCount) {
    float currentItem;
    struct Bucket *bucket;
    float bucketLimit = 0.1 * bucketCount;
    for (int i = 0; i < size; i++) {
        currentItem = floatArrayToSort[i];
        if (currentItem < 0) {
            freeBuckets(buckets, bucketCount);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 0; // No negative numbers allowed
        }
        if (currentItem < bucketLimit) {
            bucket = &(buckets[(int) (currentItem * 10)]);
        } else { // If larger than limit, store in the final bucket
            bucket = &buckets[bucketCount-1];
        }
        bucket->count++;
        if ((int) bucket->value == -1) {
            bucket->value = currentItem;
            continue;
        }
        while (bucket->next != NULL) {
            bucket = bucket->next;
        }
        struct Bucket *newBucket = (struct Bucket *)
                malloc(sizeof(struct Bucket));
        if (newBucket == NULL) {
            freeBuckets(buckets, bucketCount);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 0;
        }
        bucket->next = newBucket;
        newBucket->value = currentItem;
        newBucket->next = NULL;
        newBucket->count = 0;
    }
    return 1;
}

void mergesort_parallel(float* floatArrayToSort, int size, MPI_Comm communicator) {
    int start, portion, remainder, rank, processorCount;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processorCount);
    if (processorCount == 1) { // sequential
        mergesort(floatArrayToSort, 0, size - 1);
        return;
    }

    portion = size / processorCount;
    remainder = size % processorCount;
    getDistributions(&start, &portion, remainder, rank);

    int someRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &someRank);

    printf("Rank: %d, portion: %d, start: %d\n", someRank, portion, start);


    float sortedArray[portion];
    memcpy(sortedArray, floatArrayToSort+start, portion*sizeof(float));
    mergesort(sortedArray, 0, portion);
    if (!rank) { // Only rank 0 manages the global communications
        // Compute counts and displacements for MPI_Gatherv.
        int recvcounts[processorCount];
        int gatherDispls[processorCount]; // Displacement for MPI_Gatherv
        // rank 0 always accounts for remainder
        int valuesWithRemainder = portion * size;
        int valuesWithoutRemainder = remainder ?
                (portion-1)*size : valuesWithRemainder;

        int currGatherDispls = 0; // Start at 0
        // Processes dealing with remainders have additional counts and displs
        for (int i = 0; i < remainder; i++) {
            gatherDispls[i] = currGatherDispls;
            recvcounts[i] = valuesWithRemainder;
            currGatherDispls += valuesWithRemainder;
        }
        for (int i = remainder; i < processorCount; i++) {
            gatherDispls[i] = currGatherDispls;
            recvcounts[i] = valuesWithoutRemainder;
            currGatherDispls += valuesWithoutRemainder;
        }
        MPI_Gatherv(MPI_IN_PLACE, portion, MPI_FLOAT, floatArrayToSort,
                    recvcounts, gatherDispls, MPI_FLOAT, 0, communicator);
    }
    else {
        MPI_Gatherv(floatArrayToSort, portion, MPI_FLOAT, NULL,
                    NULL, NULL, MPI_FLOAT, 0, communicator);
    }
    // mergesort(floatArrayToSort, 0, size-1);
}

void mergesort(float* array, int low, int high) {
    if (low >= high) return;
    int mid = low + (high - low)/2;
    mergesort(array, low, mid); // low -> mid inclusive
    mergesort(array, mid + 1, high);
    merge(array, low, mid, high);
}

void merge(float* floatArrayToSort, int low, int mid, int high) {
    int i, j, k;
    int lengthOfA = mid - low + 1; // low -> mid, inclusive
    int lengthOfB = high - mid;
    float a[lengthOfA], b[lengthOfB];
    for (i = 0; i < lengthOfA; i++) {
        a[i] = floatArrayToSort[i + low];
    }
    for (j = 0; j < lengthOfB; j++) {
        b[j] = floatArrayToSort[j + mid + 1];
    }

    i = j = 0;
    k = low;
    while (i < lengthOfA && j < lengthOfB) {
        if (a[i] <= b[j]) {
            floatArrayToSort[k] = a[i];
            i++;
        }
        else {
            floatArrayToSort[k] = b[j];
            j++;
        }
        k++;
    }
    for (;i < lengthOfA; i++) {
        floatArrayToSort[k] = a[i];
        k++;
    }
    for (;j < lengthOfB; j++) {
        floatArrayToSort[k] = b[j];
        k++;
    }
}

void makeGathervCall(void* recvbuf, int rank, int processorCount, int portion,
                     int remainder, int size, MPI_Comm communicator) {
    if (!rank) { // Only rank 0 manages the global communications
        // Compute counts and displacements for MPI_Gatherv.
        int recvcounts[processorCount];
        int gatherDispls[processorCount]; // Displacement for MPI_Gatherv
        // rank 0 always accounts for remainder
        int valuesWithRemainder = portion * size;
        int valuesWithoutRemainder = remainder ?
                                     (portion-1)*size : valuesWithRemainder;

        int currGatherDispls = 0; // Start at 0
        // Processes dealing with remainders have additional counts and displs
        for (int i = 0; i < remainder; i++) {
            gatherDispls[i] = currGatherDispls;
            recvcounts[i] = valuesWithRemainder;
            currGatherDispls += valuesWithRemainder;
        }
        for (int i = remainder; i < processorCount; i++) {
            gatherDispls[i] = currGatherDispls;
            recvcounts[i] = valuesWithoutRemainder;
            currGatherDispls += valuesWithoutRemainder;
        }
        MPI_Gatherv(MPI_IN_PLACE, portion, MPI_FLOAT, recvbuf,
                    recvcounts, gatherDispls, MPI_FLOAT, 0, communicator);

    }
    else {
        MPI_Gatherv(recvbuf, portion, MPI_FLOAT, NULL,
                    NULL, NULL, MPI_FLOAT, 0, communicator);
    }
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 25000;
    float* array = (float*) malloc((size_t) size * sizeof(float));
    if (array == NULL) return -1;
//    for (int i = 0; i < size; i++) {
//        array[i] = (float) i / 10;
//    }
//    array[1] = 0.21; array[2] = 0.2;
//    array[13] = 1.4; array[14] = 1.3;
//    for (int i = 15; i < size; i++) {
//        array[39-i] = 0.1 + 0.001*i;
//    }
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        array[i] = (float) rand() / (float) RAND_MAX;
    }
    int incorrectCounter, correctCounter;
    incorrectCounter = correctCounter = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i-1]) incorrectCounter++;
        else correctCounter++;
    }
    correctCounter++; // final unaccounted number
//    printf("Initially Sorted numbers: %d\nInitially unsorted numbers: %d\n"
//           "Total numbers: %d\n", correctCounter, incorrectCounter, size);
    double start, end;
    start = MPI_Wtime();
    if (!bucketsort(array, size, 8)) { // not manager processor results
        MPI_Finalize();
        free(array);
        return 0;
    }
    end = MPI_Wtime();
    printf("Time taken: %f\n", end-start);
    MPI_Finalize();
    incorrectCounter = correctCounter = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i-1]) incorrectCounter++;
        else correctCounter++;
    }
    correctCounter++; // final unaccounted number
    printf("Sorted numbers: %d\nIncorrectly sorted numbers: %d\nTotal numbers: %d\n",
           correctCounter, incorrectCounter, size);
    free(array);
    return 0;
}