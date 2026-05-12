
# MLOps Project: MNIST End-to-End Deployment

## **Homework 2: Serving a Machine Learning Model as an API Service**

### **1. Objective & Model Selection**

* **Model:** A **Convolutional Neural Network (CNN)** for MNIST digit classification, developed originally for Homework 1.
* **Framework:** BentoML (v1.4+)

### **2. Model-Serving Configuration**

The service is defined in `service.py` using a class-based structure to handle model loading and real-time inference logic.

* **Accuracy:** The model was trained using a target-accuracy loop, stopping only after exceeding **98.7% accuracy** on the test set.
* **Preprocessing:** The service includes an automated image pipeline using `Pillow` to convert incoming images to grayscale, **invert colors** and resize the input to $28 \times 28$. The expected input image is a black number on a white background.

### **3. Local Client Request Example**

To test the service locally, ensure the BentoML server is running (`bentoml serve service:MNISTService`) and use the following `curl` command with a path to a PNG image:

```bash
curl -X POST -F "img=@seven.png" http://localhost:3000/predict
```

**Request:** A PNG file containing a handwritten "7".

**Response:** `{"The model predicts this number is: 7"}`

---

## **Homework 3: Cloud Deployment on AWS EC2**

### **1. Deployment Strategy**

* **Cloud Provider:** **AWS (EC2)**.
* **Instance Type:** `m7i-flex.large` running **Ubuntu 26.04**.
* **Deployment Method:** **Option B (Build Directly on VM)**. The project repository was cloned onto the instance, and the Docker image was built locally on the machine

### **2. Containerization**

The application was containerized using BentoML’s native Docker integration.


### **3. Launch & Configuration**

* **Networking:** AWS Security Groups were configured to open **Port 3000** for HTTP traffic and **Port 22** for SSH.
* **Docker Execution:**

```bash
docker run -d --name mnist-service -p 3000:3000 mnist_classifier:[UNIQUE_TAG]
```


### **4. Request example**

Use the following `curl` command with a path to a PNG image:

```bash
curl -X POST -F "img=@seven.png" http://13.60.94.246:3000/predict
```

**Request:** A PNG file containing a handwritten "7".

**Response:** `{"The model predicts this number is: 7"}`


### **5. Technical Tools Summary**

* **Deep Learning:** PyTorch (CNN Architecture)
* **Serving:** BentoML (Class-based API)
* **Infrastructure:** AWS EC2, Amazon EBS (gp3)
* **DevOps:** Docker, Git, cURL

---

