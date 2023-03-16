#include <stdio.h>
#include <stdlib.h>

void merge(int* intArrayToSort, int low, int mid, int high);

void mergeSort(int* intArrayToSort, int low, int high) {
    if (low >= high) return;
    int mid = low + (high - low)/2;
    mergeSort(intArrayToSort, low, mid); // low -> mid inclusive
    mergeSort(intArrayToSort, mid + 1, high);
    merge(intArrayToSort, low, mid, high);
}

void merge(int* intArrayToSort, int low, int mid, int high) {
    int i, j, k;
    int lengthOfA = mid - low + 1; // low -> mid, inclusive
    int lengthOfB = high - mid;
    int a[lengthOfA], b[lengthOfB];
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

//int compareInts(const void* a, const void* b) {
//    return (*(int*)a - *(int*)b);
//}

int main() {
//    int array[7] = {4, 0, 2, 3, 6, 5, 1};
    int array[8] = {4, 0, 2, 3, 6, 5, 7, 1};
//    mergeSort(array, 0, 6);
    mergeSort(array, 0, 7);
//    qsort(array, 8, sizeof(int), compareInts);
    for (int i = 0; i < 8; i++) {
        printf("%d\n", array[i]);
    }
    return 0;
}