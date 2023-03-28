#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

struct Bucket {
    float value;
    struct Bucket* next;
    int count;
};

void bucketsort_parallel(float* floatArrayToSort, int size,
                         int bucketCount, int itemsPerProcessor);
void getDistributions(int *start, int *portion, int remainder, int rank);
void initialiseBuckets(struct Bucket* buckets, int bucketCount);
void fillBuckets(const float* floatArrayToSort, int size,
                 struct Bucket* buckets, int bucketCount);
void printBuckets(struct Bucket* buckets, int bucketCount);
void mergesort_parallel(float* floatArrayToSort, int size, MPI_Comm communicator);
void mergesort(float* floatArrayToSort, int low, int high);
void merge(float* floatArrayToSort, int low, int mid, int high);
void makeGathervCall(void* recvbuf, int rank, int processorCount, int portion,
                     int remainder, int size, MPI_Comm communicator);


void bucketsort(float* floatArrayToSort, int size, int itemsPerProcessor) {
    bucketsort_parallel(floatArrayToSort, size, 2, itemsPerProcessor);
}

void bucketsort_parallel(float* floatArrayToSort, int size,
                         int bucketCount, int itemsPerProcessor) {
    /*
     * floatArrayToSort: array to sort, self-descriptive
     * size: size of the array to sort
     * bucketCount: number of buckets
     * itemsPerProcessor: {bucketCount} additional processors allowed for every ..-
     * -.. extra {itemsPerProcessor} items in a bucket to allow parallel mergesort
     */
    if (bucketCount < 1) return;
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
        return;
    }

    struct Bucket* buckets = (struct Bucket*) malloc((size_t) bucketCount * sizeof(struct Bucket));
    if (buckets == NULL) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return;
    }

    int portion, remainder, start, groupRank;
    groupRank = rank % bucketCount;
    portion = bucketCount / processorCount;
    remainder = bucketCount % processorCount;

    getDistributions(&start, &portion, remainder, groupRank);

    initialiseBuckets(buckets, bucketCount);
    fillBuckets(floatArrayToSort, size, buckets, bucketCount);

    if (rank) { // Rank 0 keeps original array for to have space for final gather
        size_t newSize = portion * sizeof(struct Bucket);
        memcpy(buckets, buckets + start, newSize);
        buckets = (struct Bucket *) realloc(buckets, newSize);
        if (buckets == NULL) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return;
        }
    }
    /* Create a new communicator for each bucket for parallel internal sort */
    MPI_Comm localCommunicator;
    MPI_Comm_split(MPI_COMM_WORLD, groupRank, rank, &localCommunicator);
    /* Gather items in each bucket and store in a separate array */
    int itemsInBucket;
    struct Bucket* currentBucket;
    for (int i = 0; i < portion; i++) {
        currentBucket = &buckets[i];
        itemsInBucket = currentBucket->count;
        if (itemsInBucket < 2) continue; // Single item or less in bucket
        float numbers[itemsInBucket];
        for (int j = 0; j < itemsInBucket; j++) {
            numbers[j] = currentBucket->value;
            currentBucket = currentBucket->next;
        }
        mergesort_parallel(numbers, itemsInBucket, localCommunicator);
        currentBucket = &buckets[i];
        for (int j = 0; j < itemsInBucket; j++) {
            currentBucket->value = numbers[j];
            currentBucket = currentBucket->next;
        }
    }
    makeGathervCall(buckets, rank, processorCount, portion,
                    remainder, size, MPI_COMM_WORLD);
    if (!rank) printBuckets(buckets, portion);
    /*
     * Create new communicators for each bucket.
     * This allows multiple processors to be run on each.
     * Hence, allowing mergesort to be done in parallel on each bucket.
     * The new communicators allow processor ranks to be local.
     * This allows for rank based parallelism.
     *
     * Add a "round" parameter to mergesort function
     * divide work across processors by checking if their rank is below or .--
     * -.. =above 2 to the power of "round", until there is 1 processor left
     * have the above only be done if there is 1.5x processors the number of buckets
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

void fillBuckets(const float* floatArrayToSort, int size, struct Bucket* buckets, int bucketCount) {
    float currentItem;
    struct Bucket *bucket;
    float bucketLimit = 0.1 * bucketCount;
    for (int i = 0; i < size; i++) {
        currentItem = floatArrayToSort[i];
        if (currentItem < bucketLimit) {
            bucket = &(buckets[(int) (currentItem * 10)]);
        } else { // If larger than limit, store in the final bucket
            bucket = &buckets[bucketCount-1];
        }
        bucket->count++;
        if ((int) bucket->value == -1) {
            bucket->value = currentItem;
        } else {
            while (bucket->next != NULL) {
                bucket = bucket->next;
            }
            struct Bucket *newBucket = (struct Bucket *)
                    malloc(sizeof(struct Bucket));
            bucket->next = newBucket;
            newBucket->value = currentItem;
            newBucket->next = NULL;
        }
    }
}

void printBuckets(struct Bucket* buckets, int bucketCount) {
    struct Bucket* currentBucket;
    int currentCount;
    for (int i = 0; i < bucketCount; i++) {
        currentBucket = &buckets[i];
        currentCount = currentBucket->count;
        if (currentCount < 1) continue;
        for (int j = 0; j < currentCount-1; j++) {
            printf("%f\n", currentBucket->value);
            currentBucket = currentBucket->next;
        }
        printf("%f\n", currentBucket->value);
    }
}

void mergesort_parallel(float* floatArrayToSort, int size, MPI_Comm communicator) {
    int start, portion, remainder, rank, processorCount;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &processorCount);
    portion = size / processorCount;
    remainder = size % processorCount;
    getDistributions(&start, &portion, remainder, rank);

    if (processorCount == 1) { // sequential
        mergesort(floatArrayToSort, start, size - 1);
        return;
    }

    float sortedArray[portion];
    memcpy(sortedArray, floatArrayToSort+start, portion*sizeof(float));
    mergesort(sortedArray, start, start+portion);
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

void mergesort(float* floatArrayToSort, int low, int high) {
    if (low >= high) return;
    int mid = low + (high - low)/2;
    mergesort(floatArrayToSort, low, mid); // low -> mid inclusive
    mergesort(floatArrayToSort, mid + 1, high);
    merge(floatArrayToSort, low, mid, high);
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
    int size = 15;
    float* array = (float*) malloc((size_t) size * sizeof(float));
    if (array == NULL) return -1;
    for (int i = 0; i < size; i++) {
        array[i] = (float) i / 10;
    }
    array[1] = 0.21; array[2] = 0.2;
    array[13] = 1.4; array[14] = 1.3;
    bucketsort(array, size, 8);
    MPI_Finalize();
    free(array);
    return 0;
}