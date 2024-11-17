# **Brain Tumor Detection**

This repository implements a deep learning-based brain tumor detection system using Convolutional Neural Networks (CNNs). The project includes a Flask web application for real-time MRI image classification and a CI/CD pipeline for streamlined development, deployment, and reproducibility.

---

## **Features**
- **CNN-based Detection:** Accurate classification of brain tumor types from MRI images using deep learning models.
- **Flask Application:** User-friendly web interface for uploading MRI images and obtaining predictions in real-time.
- **CI/CD Pipeline:** Automated workflows leveraging Docker and DVC for efficient development and deployment processes.
- **Modular Pipeline Structure:** Components for data ingestion, preprocessing, training, evaluation, and deployment are organized for scalability and maintainability.

---

## **Technologies Used**
- Deep Learning Frameworks: TensorFlow/Keras
- Web Framework: Flask
- Pipeline Tools: DVC, Docker
- CI/CD: GitHub Actions
- Versioning: Git

## **Workflows**

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline  
8. Update the main.py
9. Update the dvc.yaml
