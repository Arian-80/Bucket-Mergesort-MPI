#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct Bucket {
    float value;
    struct Bucket* next;
};

void bucketsort(const float* floatArrayToSort, int size);
void initialiseBuckets(struct Bucket* buckets, int bucketCount);
void printBuckets(struct Bucket* buckets, int bucketCount);
void fillBuckets(const float* floatArrayToSort, int size, struct Bucket* buckets, int bucketCount);

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

    /* Allocate memory for respective portion of buckets */
    struct Bucket* uniqueBuckets;
    uniqueBuckets = (struct Bucket*) malloc((size_t) portion * sizeof(struct Bucket));
    if (uniqueBuckets == NULL) {
        MPI_Abort(MPI_COMM_WORLD, -1);
        return;
    }

    /* Copy the portion on to the newly assigned memory space */
    for (int i = 0; i < portion; i++) {
        uniqueBuckets[i] = buckets[i+start];
    }
    free(buckets);
}

void bucketsort(const float* floatArrayToSort, int size) {
    bucketsort_parallel(floatArrayToSort, size, 10);
}

void initialiseBuckets(struct Bucket* buckets, int bucketCount) {
    for (int i = 0; i < bucketCount; i++) {
        buckets[i].value = -1;
        buckets[i].next = NULL;
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
    for (int i = 0; i < bucketCount; i++) {
        currentBucket = &buckets[i];
        while (currentBucket->next != NULL) {
            printf("%f\n", currentBucket->value);
            currentBucket = currentBucket->next;
        }
        printf("%f\n", currentBucket->value);
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
    bucketsort(array, size);
    MPI_Finalize();
    return 0;
}