#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784      // 28x28
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 100     // 0-9
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define LEARNING_RATE 0.3
#define EPOCHS 10

#define TIMEIT(label, code) { \
    clock_t _st = clock(); code; \
    printf("%s took %.0f ms\n", #label, (double)(clock() - _st) / CLOCKS_PER_SEC * 1000.0); \
}


double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double d_sigmoid(double x)
{
    return x * (1 - x);
}

void init_weights(double *weights, int size)
{
    for (int i = 0; i < size; i++)
    {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

void read_csv(const char *filename, double **images, int *labels, int size)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    char line[8192];
    for (int i = 0; i < size; i++)
    {
        if (fgets(line, sizeof(line), file))
        {
            char *token = strtok(line, ",");
            labels[i] = atoi(token);

            for (int j = 0; j < INPUT_SIZE; j++)
            {
                token = strtok(NULL, ",");
                images[i][j] = (double)atoi(token);
            }
        }
    }

    fclose(file);
}

void normalize_data(double **images, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            images[i][j] /= 255.0;
        }
    }
}

void forward(double *input, double *h_weights, double *o_weights, double *h_layer, double *o_layer) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_layer[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_layer[i] += input[j] * h_weights[i * INPUT_SIZE + j];
        }
        h_layer[i] = sigmoid(h_layer[i]);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        o_layer[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            o_layer[i] += h_layer[j] * o_weights[i * HIDDEN_SIZE + j];
        }
        o_layer[i] = sigmoid(o_layer[i]);
    }
}

void backprop(double *input, int label, double *h_weights, double *o_weights, double *h_layer, double *o_layer) {
    double o_error[OUTPUT_SIZE];
    double h_error[HIDDEN_SIZE];

    // out
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double target = (i == label) ? 1.0 : 0.0;
        o_error[i] = (target - o_layer[i]) * d_sigmoid(o_layer[i]);
    }

    // hidden
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_error[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            h_error[i] += o_error[j] * o_weights[j * HIDDEN_SIZE + i];
        }
        h_error[i] *= d_sigmoid(h_layer[i]);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            o_weights[i * HIDDEN_SIZE + j] += LEARNING_RATE * o_error[i] * h_layer[j];
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_weights[i * INPUT_SIZE + j] += LEARNING_RATE * h_error[i] * input[j];
        }
    }
}


void train_net(double **images, int *labels, double *h_weights, double *o_weights) {
    double h_layer[HIDDEN_SIZE];
    double o_layer[OUTPUT_SIZE];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < TRAIN_SIZE; i++) {

            forward(images[i], h_weights, o_weights, h_layer, o_layer);
            // mse
            double loss = 0.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                double target = (j == labels[i]) ? 1.0 : 0.0;
                loss += pow(target - o_layer[j], 2);
            }
            total_loss += loss;

            backprop(images[i], labels[i], h_weights, o_weights, h_layer, o_layer);
        }

        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / TRAIN_SIZE);
    }
}

void test_net(double **test_images, int *test_labels, double *h_weights, double *o_weights) {
    double h_layer[HIDDEN_SIZE];
    double o_layer[OUTPUT_SIZE];
    int correct_pred = 0;

    for (int i = 0; i < TEST_SIZE; i++) {
        forward(test_images[i], h_weights, o_weights, h_layer, o_layer);

        int pred_label = 0;
        double max_output = o_layer[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (o_layer[j] > max_output) {
                max_output = o_layer[j];
                pred_label = j;
            }
        }
        if (pred_label == test_labels[i]) {
            correct_pred++;
        }
    }

    double accuracy = (double)correct_pred / TEST_SIZE * 100.0;
    printf("Test Accuracy: %.2f%%\n", accuracy);
}

// print first n images and labels
void print_data(unsigned char **images, int *labels, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("label: %d\n", labels[i]);
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            printf("%3d ", images[i][j]);
            if ((j + 1) % 28 == 0)
                printf("\n");
        }
        printf("\n\n");
    }
}

int main()
{
    srand(1);

    double **train_images = (double **)malloc(TRAIN_SIZE * sizeof(double *));
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        train_images[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    }

    int *train_labels = (int *)malloc(TRAIN_SIZE * sizeof(int));

    double **test_images = (double **)malloc(TEST_SIZE * sizeof(double *));
    for (int i = 0; i < TEST_SIZE; i++)
    {
        test_images[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    }

    int *test_labels = (int *)malloc(TEST_SIZE * sizeof(int));

    TIMEIT(load_and_norm,
        read_csv("mnist_test.csv", test_images, test_labels, TEST_SIZE);
        normalize_data(test_images, TEST_SIZE);
        read_csv("mnist_train.csv", train_images, train_labels, TRAIN_SIZE);
        normalize_data(train_images, TRAIN_SIZE);
    );

    double h_weights[HIDDEN_SIZE * INPUT_SIZE];
    double o_weights[OUTPUT_SIZE * HIDDEN_SIZE];

    init_weights(h_weights, HIDDEN_SIZE * INPUT_SIZE);
    init_weights(o_weights, OUTPUT_SIZE * HIDDEN_SIZE);

    printf("Training\n");
    TIMEIT(train, train_net(train_images, train_labels, h_weights, o_weights));
    TIMEIT(test, test_net(test_images, test_labels, h_weights, o_weights));

    for (int i = 0; i < TRAIN_SIZE; i++) {
        free(train_images[i]);
    }
    free(train_images);

    for (int i = 0; i < TEST_SIZE; i++) {
        free(test_images[i]);
    }
    free(test_images);
    free(train_labels);
    free(test_labels);
    
    return 0;
}
