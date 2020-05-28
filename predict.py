{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyMtJm03X48/8BlG+MIt1J49",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/DnzDmn/Image_Classifier/blob/master/predict.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Wd68Ob7N_XeM",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 212
        },
        "outputId": "9c8c4355-39c5-4099-8cd1-d7583e31cd1b"
      },
      "source": [
        "!pip install -q -U \"tensorflow-gpu==2.0.0b1\"\n",
        "!pip install -q -U tensorflow_hub\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "import pandas as pd\n",
        "import tensorflow as tf\n",
        "import tensorflow_hub as hub\n",
        "import logging\n",
        "import argparse\n",
        "import sys\n",
        "import json\n",
        "from PIL import Image\n",
        "\n",
        "logger = tf.get_logger()\n",
        "logger.setLevel(logging.ERROR)\n",
        "parser = argparse.ArgumentParser ()\n",
        "parser.add_argument ('--image_dir', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image.', type = str)\n",
        "parser.add_argument('--model', help='Trained Model.', type=str)\n",
        "parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes.', type = int)\n",
        "parser.add_argument ('--classes' , default = 'label_map.json', help = 'Mapping of categories to real names.', type = str)\n",
        "commands = parser.parse_args()\n",
        "image_path = commands.image_dir\n",
        "model_path = commands.model\n",
        "top_k = commands.top_k\n",
        "classes = commands.classes\n",
        "image_size = 224\n",
        "\n",
        "# Creating the process_image function\n",
        "with open(classes, 'r') as f:\n",
        "    class_names = json.load(f)\n",
        "\n",
        "#Load Model\n",
        "model = tf.keras.models.load_model(\n",
        "  model_path, \n",
        "  custom_objects={'KerasLayer': hub.KerasLayer})\n",
        "\n",
        "def process_image(img):\n",
        "    image = np.squeeze(img)\n",
        "    image = tf.image.resize(image, (image_size, image_size))/255.0\n",
        "    return image\n",
        "\n",
        "\n",
        "#Prediction\n",
        "\n",
        "def predict(image_path, model, top_k = 5):\n",
        "    im = Image.open(image_path)\n",
        "    image = np.asarray(im)\n",
        "    image = process_image(image)\n",
        "    \n",
        "    prediction = model.predict(np.expand_dims(image, 0))\n",
        "\n",
        "    dataframe = pd.DataFrame(prediction[0]).reset_index().rename(columns = {'index': 'class_code', \n",
        "                                                            0: 'prob'})\\\n",
        "                        .sort_values(by='prob', ascending = False).head(top_k).reset_index(drop=True)\n",
        "    dataframe.loc[:,'class_code'] = dataframe.class_code + 1\n",
        "    dataframe.loc[:,'class_name'] = np.nan\n",
        "\n",
        "    for value in dataframe['class_code'].values:\n",
        "        class_name = class_names[str(value)]\n",
        "        dataframe.loc[:,'class_name'] = np.where(dataframe.class_code==value, class_name, dataframe.class_name)\n",
        "    \n",
        "    class_list = []\n",
        "    prob_list = []\n",
        "    \n",
        "    for class_name, prob in dataframe[['class_name','prob']].values:\n",
        "        class_list.append(class_name)\n",
        "        prob_list.append(prob)\n",
        "           \n",
        "   \n",
        "    print(\"Best Classes: {}\".format(class_list))\n",
        "    print(\"Probs: {}\".format(prob_list))\n",
        "    return dataframe, image\n",
        "\n",
        "\n",
        "\n",
        "if __name__ == \"__main__\":a\n",
        "    predict(image_path,model,top_k)"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[K     |████████████████████████████████| 348.9MB 49kB/s \n",
            "\u001b[K     |████████████████████████████████| 3.1MB 40.5MB/s \n",
            "\u001b[K     |████████████████████████████████| 501kB 47.0MB/s \n",
            "\u001b[?25h"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "usage: ipykernel_launcher.py [-h] [--image_dir IMAGE_DIR] [--model MODEL]\n",
            "                             [--top_k TOP_K] [--classes CLASSES]\n",
            "ipykernel_launcher.py: error: unrecognized arguments: -f /root/.local/share/jupyter/runtime/kernel-b4c494ce-e0ba-489e-82a5-0faba307f0c1.json\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "error",
          "ename": "SystemExit",
          "evalue": "ignored",
          "traceback": [
            "An exception has occurred, use %tb to see the full traceback.\n",
            "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 2\n"
          ]
        }
      ]
    }
  ]
}