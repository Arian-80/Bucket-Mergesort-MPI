#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

struct Bucket {
    float value;
    struct Bucket* next;
    int count;
};

void bucketsort_parallel(const float* floatArrayToSort, int size, int bucketCount);
void initialiseBuckets(struct Bucket* buckets, int bucketCount);
void fillBuckets(const float* floatArrayToSort, int size, struct Bucket* buckets, int bucketCount);
void printBuckets(struct Bucket* buckets, int bucketCount);
void mergeSort(float* floatArrayToSort, int low, int high);
void merge(float* floatArrayToSort, int low, int mid, int high);

void bucketsort(const float* floatArrayToSort, int size) {
    bucketsort_parallel(floatArrayToSort, size, 10);
}

void bucketsort_parallel(const float* floatArrayToSort, int size, int bucketCount) {
    if (bucketCount < 1) return;
    int processorCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processorCount);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (bucketCount < processorCount) { // At most 1 bucket per processor
        if (!rank)
            printf("Number of buckets is smaller than the number of"
                   " processors (%d).\nEither increase the number of buckets or "
                   "decrease the number of processors.\n", bucketCount, processorCount);
        return;
    }

    struct Bucket* buckets = (struct Bucket*) malloc((size_t) bucketCount * sizeof(struct Bucket));
    if (buckets == NULL) return;

    initialiseBuckets(buckets, bucketCount);
    fillBuckets(floatArrayToSort, size, buckets, bucketCount);

    int remainder, portion;
    remainder = bucketCount % processorCount;
    portion = bucketCount / processorCount;

    int start;
    if (rank < remainder) { // Spread remainder evenly across processors
        portion++; // Processors get a max 1 remainder bucket to deal with
        start = portion*rank; // All processors before have had remainder buckets
    }
    else {
        start = (portion+1)*remainder + portion*(rank-remainder);
    }

    if (rank) { // Rank 0 keeps original array for to have space for final gather
        size_t newSize = portion * sizeof(struct Bucket);
        memcpy(buckets, buckets + start, newSize);
        buckets = (struct Bucket *) realloc(buckets, newSize);
    }
    /* Gather items in each bucket and store in a separate array */
    int itemsInBucket;
    struct Bucket* currentBucket;
    for (int i = 0; i < portion; i++) {
        currentBucket = &buckets[i];
        itemsInBucket = currentBucket->count;
        float numbers[itemsInBucket];
        for (int j = 0; j < itemsInBucket; j++) {
            numbers[j] = currentBucket->value;
            currentBucket = currentBucket->next;
        }
        mergeSort(numbers, 0, itemsInBucket-1);
        currentBucket = &buckets[i];
        for (int j = 0; j < itemsInBucket; j++) {
            currentBucket->value = numbers[j];
            currentBucket = currentBucket->next;
        }
    }
    printBuckets(buckets, portion);
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
            struct Bucket *newBucket = (struct Bucket *) malloc(sizeof(struct Bucket));
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


void mergeSort(float* floatArrayToSort, int low, int high) {
    if (low >= high) return;
    int mid = low + (high - low)/2;
    mergeSort(floatArrayToSort, low, mid); // low -> mid inclusive
    mergeSort(floatArrayToSort, mid + 1, high);
    merge(floatArrayToSort, low, mid, high);
}

void merge(float* intArrayToSort, int low, int mid, int high) {
    int i, j, k;
    int lengthOfA = mid - low + 1; // low -> mid, inclusive
    int lengthOfB = high - mid;
    float a[lengthOfA], b[lengthOfB];
    for (i = 0; i < lengthOfA; i++) {
        a[i] = intArrayToSort[i + low];
    }
    for (j = 0; j < lengthOfB; j++) {
        b[j] = intArrayToSort[j + mid + 1];
    }

    i = j = 0;
    k = low;
    while (i < lengthOfA && j < lengthOfB) {
        if (a[i] <= b[j]) {
            intArrayToSort[k] = a[i];
            i++;
        }
        else {
            intArrayToSort[k] = b[j];
            j++;
        }
        k++;
    }
    for (;i < lengthOfA; i++) {
        intArrayToSort[k] = a[i];
        k++;
    }
    for (;j < lengthOfB; j++) {
        intArrayToSort[k] = b[j];
        k++;
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
    bucketsort(array, size);
    MPI_Finalize();
    free(array);
    return 0;
}