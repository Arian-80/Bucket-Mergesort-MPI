#include <stdio.h>
#include <stdlib.h>
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


//    if (rank) { // Rank 0 keeps full buckets array to later gather from all
        /* Allocate memory for respective portion of buckets */
        struct Bucket *uniqueBuckets;
        uniqueBuckets = (struct Bucket *) malloc((size_t) portion * sizeof(struct Bucket));
        if (uniqueBuckets == NULL) {
            MPI_Abort(MPI_COMM_WORLD, -1);
            return;
        }

        /* Copy the portion of buckets on to the newly assigned memory space */
        for (int i = 0; i < portion; i++) {
            uniqueBuckets[i] = buckets[i + start];
        }
        free(buckets);

        /* Gather items in each bucket and store in a separate array */
        int itemsInBucket;
        struct Bucket* currentBucket;
        for (int i = 0; i < portion; i++) {
            currentBucket = &uniqueBuckets[i];
            itemsInBucket = currentBucket->count;
            float numbers[itemsInBucket];
            for (int j = 0; j < itemsInBucket; j++) {
                numbers[j] = currentBucket->value;
                currentBucket = currentBucket->next;
            }
            mergeSort(numbers, 0, itemsInBucket-1);
            currentBucket = &uniqueBuckets[i];
            for (int j = 0; j < itemsInBucket; j++) {
                currentBucket->value = numbers[j];
                currentBucket = currentBucket->next;
            }
        }
        printBuckets(uniqueBuckets, portion);
//    }
    /*
     * Create new communicators for each bucket.
     * This allows multiple processors to be run on each.
     * Hence, allowing mergesort to be done in parallel on each bucket.
     * The new communicators allow processor ranks to be local.
     * This allows for rank based parallelism.
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
    bucketsort(array, size);
    MPI_Finalize();
    free(array);
    return 0;
}