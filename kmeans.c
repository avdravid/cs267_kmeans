#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

// Structure to represent a data point
typedef struct {
    double* values;  // Array of feature values
    int dimensions;  // Number of dimensions/features
    int cluster;     // Assigned cluster
} Point;

// Structure to represent a centroid
typedef struct {
    double* values;  // Array of feature values
    int dimensions;  // Number of dimensions/features
    int count;       // Number of points in this cluster
} Centroid;

// Function to calculate Euclidean distance between a point and a centroid
double calculateDistance(Point point, Centroid centroid) {
    double sum = 0.0;
    for (int i = 0; i < point.dimensions; i++) {
        double diff = point.values[i] - centroid.values[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Count the number of rows and columns in a CSV file
void getCSVDimensions(const char* filename, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    
    char line[1024];
    *rows = 0;
    *cols = 0;
    
    // Get the first line to count columns
    if (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        while (token != NULL) {
            (*cols)++;
            token = strtok(NULL, ",");
        }
        (*rows)++;
    }
    
    // Count remaining rows
    while (fgets(line, sizeof(line), file)) {
        (*rows)++;
    }
    
    fclose(file);
}

// Read data from CSV file into points array
Point* readCSVData(const char* filename, int* numPoints, int* dimensions) {
    int rows, cols;
    getCSVDimensions(filename, &rows, &cols);
    
    *numPoints = rows;
    *dimensions = cols;
    
    // Allocate memory for points
    Point* points = (Point*)malloc(rows * sizeof(Point));
    if (points == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Open the file again for reading data
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    
    char line[1024];
    int row = 0;
    
    while (fgets(line, sizeof(line), file) && row < rows) {
        // Allocate memory for this point's values
        points[row].values = (double*)malloc(cols * sizeof(double));
        points[row].dimensions = cols;
        points[row].cluster = -1;
        
        char* tmp = strdup(line);
        if (tmp == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        
        char* token = strtok(tmp, ",");
        int col = 0;
        
        while (token != NULL && col < cols) {
            points[row].values[col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        
        free(tmp);
        row++;
    }
    
    fclose(file);
    return points;
}

// Read centroids from CSV file
Centroid* readCentroidsFromCSV(const char* filename, int* k, int expectedDimensions) {
    int rows, cols;
    getCSVDimensions(filename, &rows, &cols);
    
    if (cols != expectedDimensions) {
        fprintf(stderr, "Error: Centroids dimensions (%d) don't match data dimensions (%d)\n", 
                cols, expectedDimensions);
        exit(1);
    }
    
    *k = rows;
    
    // Allocate memory for centroids
    Centroid* centroids = (Centroid*)malloc(rows * sizeof(Centroid));
    if (centroids == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Open the file again for reading data
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    
    char line[1024];
    int row = 0;
    
    while (fgets(line, sizeof(line), file) && row < rows) {
        // Allocate memory for this centroid's values
        centroids[row].values = (double*)malloc(cols * sizeof(double));
        centroids[row].dimensions = cols;
        centroids[row].count = 0;
        
        char* tmp = strdup(line);
        if (tmp == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        
        char* token = strtok(tmp, ",");
        int col = 0;
        
        while (token != NULL && col < cols) {
            centroids[row].values[col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        
        free(tmp);
        row++;
    }
    
    fclose(file);
    return centroids;
}

// Function to initialize centroids using random selection from the dataset
void initializeCentroids(Point* points, int numPoints, Centroid* centroids, int k) {
    // Simple initialization: evenly spaced points from the dataset
    // In practice, k-means++ would be better
    int step = numPoints / k;
    if (step == 0) step = 1;
    
    for (int i = 0; i < k; i++) {
        int idx = (i * step) % numPoints;
        centroids[i].values = (double*)malloc(points[idx].dimensions * sizeof(double));
        centroids[i].dimensions = points[idx].dimensions;
        centroids[i].count = 0;
        
        for (int j = 0; j < points[idx].dimensions; j++) {
            centroids[i].values[j] = points[idx].values[j];
        }
    }
}

// Function to assign each point to the nearest centroid
int assignPointsToClusters(Point* points, int numPoints, Centroid* centroids, int k) {
    int changes = 0;
    
    for (int i = 0; i < numPoints; i++) {
        double minDistance = DBL_MAX;
        int nearestCluster = 0;
        
        for (int j = 0; j < k; j++) {
            double distance = calculateDistance(points[i], centroids[j]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = j;
            }
        }
        
        if (points[i].cluster != nearestCluster) {
            points[i].cluster = nearestCluster;
            changes++;
        }
    }
    
    return changes;
}

// Function to update the centroids based on assigned points
void updateCentroids(Point* points, int numPoints, Centroid* centroids, int k) {
    // Reset centroid values and counts
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < centroids[i].dimensions; j++) {
            centroids[i].values[j] = 0.0;
        }
        centroids[i].count = 0;
    }
    
    // Sum up all points in each cluster
    for (int i = 0; i < numPoints; i++) {
        int cluster = points[i].cluster;
        centroids[cluster].count++;
        
        for (int j = 0; j < points[i].dimensions; j++) {
            centroids[cluster].values[j] += points[i].values[j];
        }
    }
    
    // Calculate average to get new centroid positions
    for (int i = 0; i < k; i++) {
        if (centroids[i].count > 0) {
            for (int j = 0; j < centroids[i].dimensions; j++) {
                centroids[i].values[j] /= centroids[i].count;
            }
        }
    }
}

// Main K-means algorithm
void kMeans(Point* points, int numPoints, Centroid* centroids, int k, int maxIterations) {
    int iterations = 0;
    int changes;
    
    // Main loop
    do {
        // Assign points to nearest centroids
        changes = assignPointsToClusters(points, numPoints, centroids, k);
        
        // Update centroids based on assigned points
        updateCentroids(points, numPoints, centroids, k);
        
        iterations++;
        
        printf("Iteration %d: %d points changed clusters\n", iterations, changes);
        
    } while (changes > 0 && iterations < maxIterations);
    
    printf("K-means clustering completed after %d iterations\n", iterations);
    
    // Print cluster information
    for (int i = 0; i < k; i++) {
        printf("\nCluster %d:\n", i);
        printf("Center: (");
        for (int j = 0; j < centroids[i].dimensions; j++) {
            printf("%0.2f", centroids[i].values[j]);
            if (j < centroids[i].dimensions - 1) printf(", ");
        }
        printf(")\n");
        printf("Points in cluster: %d\n", centroids[i].count);
    }
}

// Print usage information
void printUsage(char* programName) {
    printf("Usage: %s <data_csv> <num_clusters> [max_iterations] [centroids_csv]\n", programName);
    printf("  <data_csv>      : Path to the CSV file containing the data points\n");
    printf("  <num_clusters>  : Number of clusters (k)\n");
    printf("  [max_iterations]: Maximum number of iterations (default: 100)\n");
    printf("  [centroids_csv] : Optional path to CSV file with initial centroids\n");
    printf("                    (if provided, num_clusters is ignored and taken from this file)\n");
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 3 || argc > 5) {
        printUsage(argv[0]);
        return 1;
    }
    
    const char* dataFilename = argv[1];
    int k = atoi(argv[2]);
    int maxIterations = 100;  // Default value
    const char* centroidsFilename = NULL;
    
    if (argc >= 4) {
        maxIterations = atoi(argv[3]);
    }
    
    if (argc == 5) {
        centroidsFilename = argv[4];
    }
    
    if (k <= 0 && centroidsFilename == NULL) {
        fprintf(stderr, "Number of clusters must be positive\n");
        return 1;
    }
    
    // Read data from CSV
    int numPoints, dimensions;
    Point* points = readCSVData(dataFilename, &numPoints, &dimensions);
    
    printf("Read %d points with %d dimensions from %s\n", numPoints, dimensions, dataFilename);
    
    // Initialize centroids
    Centroid* centroids = NULL;
    
    if (centroidsFilename != NULL) {
        // Read centroids from file
        printf("Reading initial centroids from %s\n", centroidsFilename);
        centroids = readCentroidsFromCSV(centroidsFilename, &k, dimensions);
        printf("Using %d centroids from file\n", k);
    } else {
        // Use automatic initialization
        printf("Initializing %d centroids automatically\n", k);
        centroids = (Centroid*)malloc(k * sizeof(Centroid));
        initializeCentroids(points, numPoints, centroids, k);
    }
    
    // Run K-means
    kMeans(points, numPoints, centroids, k, maxIterations);
    
    // Optional: Save results to a new CSV file
    char outputFilename[256];
    sprintf(outputFilename, "%s.clusters.csv", dataFilename);
    FILE* outputFile = fopen(outputFilename, "w");
    
    if (outputFile) {
        // Write header
        for (int i = 0; i < dimensions; i++) {
            fprintf(outputFile, "dim%d,", i+1);
        }
        fprintf(outputFile, "cluster\n");
        
        // Write data with cluster assignments
        for (int i = 0; i < numPoints; i++) {
            for (int j = 0; j < dimensions; j++) {
                fprintf(outputFile, "%f,", points[i].values[j]);
            }
            fprintf(outputFile, "%d\n", points[i].cluster);
        }
        
        fclose(outputFile);
        printf("\nResults saved to %s\n", outputFilename);
    }
    
    // Save final centroids to a CSV file
    char centroidsOutputFilename[256];
    sprintf(centroidsOutputFilename, "%s.centroids.csv", dataFilename);
    FILE* centroidsOutputFile = fopen(centroidsOutputFilename, "w");
    
    if (centroidsOutputFile) {
        // Write centroids
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dimensions; j++) {
                fprintf(centroidsOutputFile, "%f", centroids[i].values[j]);
                if (j < dimensions - 1) fprintf(centroidsOutputFile, ",");
            }
            fprintf(centroidsOutputFile, "\n");
        }
        
        fclose(centroidsOutputFile);
        printf("Final centroids saved to %s\n", centroidsOutputFilename);
    }
    
    // Clean up
    for (int i = 0; i < numPoints; i++) {
        free(points[i].values);
    }
    free(points);
    
    for (int i = 0; i < k; i++) {
        free(centroids[i].values);
    }
    free(centroids);
    
    return 0;
}