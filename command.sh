#!/bin/bash

kind delete cluster
kind create cluster --config kind-config.yaml
kubectl create namespace onkar
kubectl apply -f kind-deployment.yml -n onkar
kubectl apply -f kind-service.yml -n onkar
kubectl get pods -n onkar -o wide
kubectl get nodes -n onkar -o wide
kubectl get svc -n onkar -o wide



