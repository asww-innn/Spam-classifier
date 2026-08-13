# SMS Spam Classifier

A machine learning project that classifies text messages as spam or legitimate ("ham"),
built with Python and scikit-learn.

## How to run it

1. Install the two libraries this needs:
   ```
   pip install pandas scikit-learn
   ```
2. Download the dataset: search **"SMS Spam Collection Dataset"** on Kaggle
   (kaggle.com/datasets/uciml/sms-spam-collection-dataset), download it, and
   save the file as `spam.csv` in the same folder as `spam_classifier.py`.
3. Run:
   ```
   python spam_classifier.py
   ```

You'll see the model's accuracy, precision, and recall printed out, followed by
predictions on a few new example messages.

## What it actually does (read this before an interview)

1. **Loads ~5,500 real text messages**, each already labelled "spam" or "ham" by humans.
2. **Splits the data** — 80% to train the model, 20% held back to test it fairly on
   messages it's never seen.
3. **Converts text to numbers using TF-IDF** — this counts which words appear in each
   message and weighs rare, distinctive words (like "WON" or "URGENT") more heavily
   than common ones (like "the" or "and"). Models can't read words directly, so this
   step is what makes the text usable.
4. **Trains a Naive Bayes classifier** — a simple, fast algorithm that learns which
   words are statistically more common in spam vs. real messages, then uses that to
   guess new messages.
5. **Evaluates itself** on the 20% of messages it never saw during training, using
   four standard metrics:
   - **Accuracy** — % of all messages classified correctly
   - **Precision** — of everything it called "spam," how much actually was spam
   - **Recall** — of all the real spam, how much did it actually catch
   - **F1 score** — a balance between precision and recall

## What to say about it

- **In one line:** "I built an SMS spam classifier in Python using scikit-learn,
  trained on ~5,500 real messages, achieving 96.9% accuracy."
- **If asked why accuracy alone isn't the full picture:** because the dataset is
  imbalanced (far more real messages than spam), a model could get high accuracy by
  just guessing "not spam" every time — that's why precision and recall matter too.
- **If asked what you'd improve:** try a different model (Logistic Regression, SVM),
  tune the TF-IDF settings, or use a larger/more recent dataset.

## Resume bullet to use

> Built an SMS spam classification model in Python using scikit-learn, achieving
> 96.9% accuracy on ~5,500 real-world messages through TF-IDF vectorization and a
> Naive Bayes classifier.
