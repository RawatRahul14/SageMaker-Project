# Data Science Projects using AWS SageMaker and S3
**This playlist consists of 2 projects using Amazon SageMaker and Amazon S3:**

The first project involves Mobile Price Classification using a Random Forest Classifier to predict mobile phone price ranges based on features, with the dataset stored in Amazon S3 and model training done on SageMaker. The second project focuses on Cat vs. Dog Image Classification using PyTorch and transfer learning, with the dataset stored in Amazon S3 and model fine-tuning performed on SageMaker.

    

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
    - [AWS CLI Installation](#aws-cli-installation)
    - [AWS Account and Permissions](#aws-account-and-permissions)
    - [AWS Configurations](#aws-configurations)
- [Datasets](#datasets)
- [Model Training with SageMaker](#model-training-with-sagemaker)
- [Running the Project](#running-the-project)
- [Conclusion](#conclusion)

## Project Overview
In these projects, you will:

- Set up an S3 bucket to store your dataset.
- Use Amazon SageMaker to build and train a Random Forest Classifier model and a Convolutional Neural Network model.
- Evaluate the model’s performance using accuracy metrics.

The primary objective is to demonstrate how cloud-based services like Amazon SageMaker and S3 can streamline the process of building, training, and deploying machine learning and deep learning models.

## Prerequisites
### AWS CLI Installation
To interact with AWS services directly from the command line, you need to install the AWS CLI.

1. Download and install the AWS CLI from the official AWS documentation:
[AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

2. After installation, you can verify it by typing:

    ```bash
    aws --version
    ```

### AWS Account and Permissions
Ensure that you have an AWS account with access to the following services:

- Amazon SageMaker for model building and training.
- Amazon S3 for storing the dataset.

You will also need to configure an IAM Role with permissions to access S3 and other necessary services like SageMaker.

### AWS Configurations
To configure the AWS CLI with your credentials:

1. Open your terminal or command prompt.
2. Run the following command:
    ```bash
    aws configure
    ```
3. Enter your AWS Access Key ID, Secret Access Key, and Region when prompted. Make sure all three are provided for successful configuration.

## Datasets
- The dataset for the Machine Learning project consists of mobile phone features and their respective price categories. The dataset includes features such as:

    - RAM
    - Battery life
    - Screen size
    - Camera quality
    - And more…

- Whereas the dataset for the Deep Learning project consists total of 5000 images of cats and dogs (1000 each for training data and 500 for validation data).

> Make an empty s3 bucket, where you can save this file.

## Model Training with SageMaker
1. Setting up the SageMaker environment:
    - Start by creating a SageMaker instance that will be used for model training.
    - Upload the dataset from your local machine to your S3 bucket.

2. Training the Models:
    - Use the Random Forest Classifier from the Scikit-learn library and resnet50 from torchvision.models.
    - Train the model on the dataset stored in S3, and evaluate its performance.
3. Evaluation:
    - After training, the model’s accuracy is evaluated using the test data.
    - Metrics like accuracy, precision, and recall are calculated to assess the model's performance.

## Running the Project
1. Set up your AWS CLI and configure your credentials.
2. Clonethe repository
    ```bash
    git clone https://github.com/RawatRahul14/SageMaker-Project.git
    ```
3. Install the packages
    ```bash
    pip install -r requirements.txt
    ```
4. Change the `name of the s3 with the name of your S3 bucket` in the codes.
5. You have to create new IAM roles to execute the project.

    Create a New IAM Role for SageMaker:
    1. Go to the IAM Console in AWS.
    2. Navigate to Roles > Create role.
    3. Choose AWS Service and select SageMaker.
    4. Attach the following policies
        - AmazonSageMakerFullAccess (essential for full SageMaker access)
    5. Name the role, e.g., SageMakerExecutionRole.
    6. Copy the ARN of the created role (e.g., arn:aws:iam::<account-id>:role/SageMakerExecutionRole).
6. Launch Amazon SageMaker to start the model training process.
7. Follow the Python script provided in the repository to interact with SageMaker.

## Conclusion
This project serves as an example of how cloud technologies like AWS SageMaker and S3 can be used for training machine learning and deep learning models. By following the steps outlined in this project, you can build, train, and evaluate a models on the cloud.