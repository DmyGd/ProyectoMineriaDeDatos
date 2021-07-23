import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator


class Kmeans_md():
    def __init__(self, data):
        self.data = data


if __name__ == "__main__":
    print("Funcionas?")
