# Project Report: Classifying Gamma-Ray Showers using Machine Learning

## 1. Introduction
For this project, I wanted to apply the concepts covered in the "Software and Computing for NSN Physics" course to a practical Astroparticle Physics problem. I chose to work with the MAGIC Gamma Telescope dataset.

The main difficulty with ground-based Cherenkov telescopes is isolating the signal from the background noise. The telescopes detect flashes of light from high-energy cosmic rays hitting the atmosphere, but the majority of these events are hadronic showers rather than the gamma-ray signals we are actually looking for. Traditionally, physicists might apply strict mathematical cuts to filter out this background. Instead, I wanted to investigate whether a Machine Learning model could automatically learn the geometric differences between the two types of showers.

## 2. Methodology

### 2.1 The Physics Data and Machine Learning (Module 1)
I sourced the dataset from the UCI Machine Learning Repository. It contains the Hillas parameters (which describe the length, width, and asymmetry of the light flashes) for thousands of simulated events.

For the analysis, I used Python and `scikit-learn`. While I initially considered using a deep neural network, I realized that for structured, tabular data with only 10 features, a Random Forest Classifier was a much better fit. It is computationally faster, less prone to overfitting, and much easier to interpret.

I used `pandas` to load the data and converted the text labels ('g' for gamma and 'h' for hadron) into binary integers so the model could process them. I then set aside 30% of the data for testing. Finally, I trained a Random Forest model with 100 decision trees to classify the events.

### 2.2 UNIX Automation (Module 2)
To streamline the pipeline, I wrote a bash script (`run_pipeline.sh`). I wanted the setup to be as automated as possible, so the script uses `wget` to download the dataset directly, removing the need for manual downloads.

I also included some basic sanity checks to run before the main Python script executes. By using `wc -l` to count the total events and `awk` to parse the CSV and count the number of gamma rays versus hadrons, I was able to quickly inspect the dataset's balance directly from the terminal.

### 2.3 Containerization (Module 3)
To ensure the project is reproducible and runs smoothly on other machines (like yours), regardless of the local Python or `scikit-learn` versions, I packaged the entire pipeline into a Docker container.

I created a `Dockerfile` that establishes a clean Linux environment, installs the exact dependencies listed in my `requirements.txt`, and automatically executes the bash script when the container is run.

## 3. Results
Upon execution, the pipeline trains the model and generates a PDF containing two main plots:
1. **The ROC Curve:** The model performed remarkably well on the unseen test data, achieving an Area Under the Curve (AUC) of approximately 0.93. This demonstrates that the Random Forest can effectively distinguish gamma rays from the hadronic background while retaining a strong signal.
2. **Feature Importances:** One of the main advantages of using a Random Forest is the ability to extract feature importances. The resulting plot shows that the model relied most heavily on `fLength` (the major axis of the shower ellipse) and `fAlpha` (the angle). This is an encouraging result, as it aligns perfectly with the underlying physics of how Cherenkov showers develop.

Overall, this project provided an excellent opportunity to bridge the machine learning techniques from Module 1 with the practical software engineering tools (UNIX and Docker) covered in Modules 2 and 3.

## 4. Bibliography
1. **MAGIC Gamma Telescope Dataset.** UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)
2. **Starmer, J. [StatQuest]. (2018).** *Random Forests Part 1 - Building, Using and Evaluating.* YouTube. [https://www.youtube.com/watch?v=J4Wdy0Wc_xQ]
3. **freeCodeCamp.org. (2022).** *Machine Learning with Python and Scikit-Learn – Full Course.* YouTube. [https://www.youtube.com/watch?v=pqNCD_5r0IU]
4. **Fermilab. (2018).** *What is Cherenkov radiation?* YouTube. (Dr. Don Lincoln explains the physics behind atmospheric Cherenkov flashes detected by MAGIC).
[https://www.youtube.com/watch?v=Yjx0BSXa0Ks]
5. **Hillas, A. M. (1985).** *Cerenkov light images of EAS produced by primary gamma rays and by nuclei.* Proceedings of the 19th International Cosmic Ray Conference (La Jolla), 3, 445-448. (The foundational physics paper defining the geometric "Hillas Parameters" used as the 10 mathematical features in this dataset).
