// Quick Sort Comparative Implementation
// Based on Mohammad Fasha's research paper
// Implementation includes all four execution modes:
// 1. Sequential
// 2. Conventional Multi-threading
// 3. OpenMP Multi-threading
// 4. MPI for Supercomputer execution

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <omp.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

// Struct to hold sub-array information for multi-threading
struct SubArrayInfo {
    int lower_bound;
    int upper_bound;
};

// Class containing all Quick Sort implementations
class QuickSorter {
private:
    // Partition function for the pivot-in-middle scheme
    int partitionMiddle(vector<int>& arr, int low, int high) {
        int pivot = arr[(low + high) / 2];
        int i = low - 1;
        int j = high + 1;
        
        while (true) {
            do {
                i++;
            } while (arr[i] < pivot);
            
            do {
                j--;
            } while (arr[j] > pivot);
            
            if (i >= j)
                return j;
                
            swap(arr[i], arr[j]);
        }
    }

public:
    // Sequential Quick Sort implementation
    void runSortPivotMiddle(vector<int>& arr, int low, int high) {
        if (low < high) {
            int pivotIndex = partitionMiddle(arr, low, high);
            runSortPivotMiddle(arr, low, pivotIndex);
            runSortPivotMiddle(arr, pivotIndex + 1, high);
        }
    }
    
    // Function to generate test arrays
    vector<int> generateArray(int size, bool sorted) {
        vector<int> arr(size);
        
        if (sorted) {
            // Generate a sorted array
            for (int i = 0; i < size; i++) {
                arr[i] = i;
            }
        } else {
            // Generate a random array
            srand(time(nullptr));
            for (int i = 0; i < size; i++) {
                arr[i] = rand();
            }
        }
        
        return arr;
    }
    
    // Sequential Quick Sort test function
    void testSequential(int size, bool sorted) {
        vector<int> arr = generateArray(size, sorted);
        
        auto start = high_resolution_clock::now();
        runSortPivotMiddle(arr, 0, size - 1);
        auto stop = high_resolution_clock::now();
        
        auto duration = duration_cast<microseconds>(stop - start);
        double seconds = duration.count() / 1000000.0;
        
        cout << "Sequential Quick Sort (" << (sorted ? "sorted" : "random") << " array of size " 
             << size << "): " << seconds << " seconds" << endl;
    }
    
    // Conventional Multi-threading Quick Sort
    void testConventionalMultiThreaded(int size, bool sorted, int numThreads) {
        vector<int> arr = generateArray(size, sorted);
        vector<thread> threads;
        vector<SubArrayInfo> subArraysInfo(numThreads);
        
        // Divide array into equal parts for each thread
        int segmentSize = size / numThreads;
        for (int i = 0; i < numThreads; i++) {
            subArraysInfo[i].lower_bound = i * segmentSize;
            subArraysInfo[i].upper_bound = (i == numThreads - 1) ? size - 1 : (i + 1) * segmentSize - 1;
        }
        
        auto start = high_resolution_clock::now();
        
        // Create threads for each segment
        for (int i = 0; i < numThreads; i++) {
            threads.push_back(thread([&arr, &subArraysInfo, i, this]() {
                QuickSorter innerSorter;
                innerSorter.runSortPivotMiddle(arr, subArraysInfo[i].lower_bound, 
                                                subArraysInfo[i].upper_bound);
            }));
        }
        
        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
        
        // Merge the sorted segments
        vector<int> mergedArray;
        for (int i = 0; i < numThreads; i++) {
            mergedArray.insert(mergedArray.end(), 
                              arr.begin() + subArraysInfo[i].lower_bound,
                              arr.begin() + subArraysInfo[i].upper_bound + 1);
        }
        
        // Final sort to ensure merged array is fully sorted
        sort(mergedArray.begin(), mergedArray.end());
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        double seconds = duration.count() / 1000000.0;
        
        cout << "Conventional Multi-threaded Quick Sort (" << (sorted ? "sorted" : "random") 
             << " array of size " << size << " with " << numThreads 
             << " threads): " << seconds << " seconds" << endl;
    }
    
    // OpenMP Multi-threading Quick Sort
    void testOpenMPMultiThreaded(int size, bool sorted, int numThreads) {
        vector<int> arr = generateArray(size, sorted);
        vector<SubArrayInfo> subArraysInfo(numThreads);
        
        // Divide array into equal parts for each thread
        int segmentSize = size / numThreads;
        for (int i = 0; i < numThreads; i++) {
            subArraysInfo[i].lower_bound = i * segmentSize;
            subArraysInfo[i].upper_bound = (i == numThreads - 1) ? size - 1 : (i + 1) * segmentSize - 1;
        }
        
        auto start = high_resolution_clock::now();
        
        // Use OpenMP to parallelize sorting
        #pragma omp parallel num_threads(numThreads)
        {
            int tid = omp_get_thread_num();
            QuickSorter innerSorter;
            innerSorter.runSortPivotMiddle(arr, subArraysInfo[tid].lower_bound, 
                                          subArraysInfo[tid].upper_bound);
        }
        
        // Merge the sorted segments
        vector<int> mergedArray;
        for (int i = 0; i < numThreads; i++) {
            mergedArray.insert(mergedArray.end(), 
                              arr.begin() + subArraysInfo[i].lower_bound,
                              arr.begin() + subArraysInfo[i].upper_bound + 1);
        }
        
        // Final sort to ensure merged array is fully sorted
        sort(mergedArray.begin(), mergedArray.end());
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        double seconds = duration.count() / 1000000.0;
        
        cout << "OpenMP Multi-threaded Quick Sort (" << (sorted ? "sorted" : "random") 
             << " array of size " << size << " with " << numThreads 
             << " threads): " << seconds << " seconds" << endl;
    }
};

// MPI Implementation for SuperComputer execution
void MPIQuickSort(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int arraySize = 5000000; // Default array size
    if (argc > 1) {
        arraySize = atoi(argv[1]);
    }
    
    if (rank == 0) {
        // Master node
        cout << "Running MPI Quick Sort with " << size << " processors and array size " 
             << arraySize << endl;
        
        // Initialize the random array
        vector<int> masterArray(arraySize);
        srand(time(nullptr));
        for (int i = 0; i < arraySize; i++) {
            masterArray[i] = rand();
        }
        
        // Time measurement start
        auto startTotal = high_resolution_clock::now();
        auto startPrep = high_resolution_clock::now();
        
        // Organize array according to pivots for better distribution
        vector<int> pivots(size - 1);
        for (int i = 0; i < size - 1; i++) {
            pivots[i] = masterArray[(i + 1) * arraySize / size];
        }
        
        // Sort the array into segments based on pivots
        vector<vector<int>> segments(size);
        for (int i = 0; i < arraySize; i++) {
            int segmentIndex = 0;
            while (segmentIndex < size - 1 && masterArray[i] > pivots[segmentIndex]) {
                segmentIndex++;
            }
            segments[segmentIndex].push_back(masterArray[i]);
        }
        
        auto stopPrep = high_resolution_clock::now();
        auto durationPrep = duration_cast<microseconds>(stopPrep - startPrep);
        double secondsPrep = durationPrep.count() / 1000000.0;
        cout << "Preparation time: " << secondsPrep << " seconds" << endl;
        
        // Send segments to worker nodes
        auto startSend = high_resolution_clock::now();
        for (int i = 1; i < size; i++) {
            int segmentSize = segments[i].size();
            MPI_Send(&segmentSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            if (segmentSize > 0) {
                MPI_Send(segments[i].data(), segmentSize, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        auto stopSend = high_resolution_clock::now();
        auto durationSend = duration_cast<microseconds>(stopSend - startSend);
        double secondsSend = durationSend.count() / 1000000.0;
        cout << "Sending time: " << secondsSend << " seconds" << endl;
        
        // Sort the master's segment
        QuickSorter sorter;
        if (!segments[0].empty()) {
            sorter.runSortPivotMiddle(segments[0], 0, segments[0].size() - 1);
        }
        
        // Receive sorted segments back
        auto startReceive = high_resolution_clock::now();
        for (int i = 1; i < size; i++) {
            int segmentSize;
            MPI_Recv(&segmentSize, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            segments[i].resize(segmentSize);
            if (segmentSize > 0) {
                MPI_Recv(segments[i].data(), segmentSize, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        auto stopReceive = high_resolution_clock::now();
        auto durationReceive = duration_cast<microseconds>(stopReceive - startReceive);
        double secondsReceive = durationReceive.count() / 1000000.0;
        cout << "Receiving time: " << secondsReceive << " seconds" << endl;
        
        // Concatenate all sorted segments
        vector<int> sortedArray;
        for (int i = 0; i < size; i++) {
            sortedArray.insert(sortedArray.end(), segments[i].begin(), segments[i].end());
        }
        
        auto stopTotal = high_resolution_clock::now();
        auto durationTotal = duration_cast<microseconds>(stopTotal - startTotal);
        double secondsTotal = durationTotal.count() / 1000000.0;
        
        cout << "Total MPI execution time: " << secondsTotal << " seconds" << endl;
        cout << "Estimated actual sort time (excluding communication): " 
             << secondsTotal - secondsPrep - secondsSend - secondsReceive << " seconds" << endl;
        
        // Verify the array is sorted
        bool sorted = true;
        for (int i = 1; i < sortedArray.size(); i++) {
            if (sortedArray[i] < sortedArray[i-1]) {
                sorted = false;
                break;
            }
        }
        cout << "Array is " << (sorted ? "sorted" : "not sorted") << endl;
    } else {
        // Worker nodes
        int segmentSize;
        MPI_Recv(&segmentSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        vector<int> segment(segmentSize);
        if (segmentSize > 0) {
            MPI_Recv(segment.data(), segmentSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Sort the received segment
            QuickSorter sorter;
            sorter.runSortPivotMiddle(segment, 0, segmentSize - 1);
            
            // Send back the sorted segment
            MPI_Send(&segmentSize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(segment.data(), segmentSize, MPI_INT, 0, 1, MPI_COMM_WORLD);
        } else {
            MPI_Send(&segmentSize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
}

// Main function to run benchmark tests
int main(int argc, char* argv[]) {
    if (argc > 1 && string(argv[1]) == "mpi") {
        // Run MPI version for SuperComputer
        MPIQuickSort(argc - 1, argv + 1);
        return 0;
    }
    
    // Array sizes to test
    vector<int> arraySizes = {1000000, 5000000, 10000000, 50000000, 100000000};
    
    QuickSorter sorter;
    int numThreads = 8; // Number of threads to use for multi-threading tests
    
    cout << "=== Quick Sort Benchmarking ===" << endl;
    
    // Run sequential tests
    cout << "\n== Sequential Quick Sort Tests ==" << endl;
    for (int size : arraySizes) {
        sorter.testSequential(size, true);  // Sorted array
        sorter.testSequential(size, false); // Random array
    }
    
    // Run conventional multi-threading tests
    cout << "\n== Conventional Multi-Threading Tests ==" << endl;
    for (int size : arraySizes) {
        sorter.testConventionalMultiThreaded(size, true, numThreads);  // Sorted array
        sorter.testConventionalMultiThreaded(size, false, numThreads); // Random array
    }
    
    // Run OpenMP multi-threading tests
    cout << "\n== OpenMP Multi-Threading Tests ==" << endl;
    for (int size : arraySizes) {
        sorter.testOpenMPMultiThreaded(size, true, numThreads);  // Sorted array
        sorter.testOpenMPMultiThreaded(size, false, numThreads); // Random array
    }
    
    cout << "\nTo run the MPI version for SuperComputer execution, use:" << endl;
    cout << "mpirun -np <number_of_processors> " << argv[0] << " mpi <array_size>" << endl;
    
    return 0;
}