# ae-studio
take-home test for data scientists

# Data Science Screening Questions

This repository contains answers and code snippets for several data science screening questions. Each question is addressed with a concise summary, key solution points, and links to the detailed code or further explanations.

---

## Question 1: Appending "!" to a List of Strings

**Short Summary:**  
Create a new list by appending "!" to each string in a list. Two methods are demonstrated: using a list comprehension and using recursion.

**Key Points of the Solution:**  
- **List Comprehension:** A concise one-line method that iterates over each string and appends "!".
- **Recursion:** A function that processes the list element by element, appending "!" and recursively handling the remainder of the list.

**Code Reference:**  
- **List Comprehension:**  
def append_exclamation_list_comp(strings):
    """Return a new list with '!' appended to each string using list comprehension."""
    return [s + "!" for s in strings]

- **Recursion:**
def append_exclamation_recursive(strings):
    """Return a new list with '!' appended to each string using recursion."""
    if not strings:  # base case: if the list is empty, return an empty list
        return []
    # recursive case: append "!" to the first string and recurse on the remainder of the list
    return [strings[0] + "!"] + append_exclamation_recursive(strings[1:])

---

## Question 2: Email Priority Classification

**Short Summary:**  
Determine the importance of understanding the `is_contact` feature when classifying email priority. The feature indicates whether the sender is in the recipient’s contact list.

**Key Points of the Solution:**  
- **Clarification on Data Timing:** In a first conversation with the client, I would ask something like:
“Does is_contact indicate the sender was in the recipient’s address book at the time the email was received, or was it updated later (e.g. if the recipient added the sender to contacts afterward)?”
--If is_contact is determined at email receipt time, then it’s a valid feature for modeling (it’s information that would genuinely be available when predicting priority for new incoming emails).
--If instead the contact list is updated over time and the dataset simply marks whether the sender is in the contact list now or at the end of the two-year period, then some emails could be marked as is_contact = True even though at the time they were received, the sender was not yet in the contact list. This would mean the model is using information that would not have been available when the email arrived. In other words, it would be leaking future knowledge (the fact that the sender was eventually added to contacts) into the training data. This could artificially inflate model performance and would not generalize to real future emails.
- **Avoiding Data Leakage:** Ensuring that the feature is measured in real-time is critical to avoid future information leaking into the training data.
- **Implications:** This clarification determines if the feature can be reliably used for training and prediction.

---

## Question 3: Online Logistic Regression with Fixed Weights

**Short Summary:**  
Outline an approach for implementing an online logistic regression model where some weights remain fixed during learning.

**Key Points of the Solution:**  
- **Model Initialization:** Define weights and identify which ones are fixed.
- **Online Updates:** Process each data point sequentially using stochastic gradient descent, updating only the trainable weights.
- **Library Support:** Options include customizing optimizers in libraries like PyTorch or TensorFlow by freezing fixed weights via flags (e.g., `requires_grad=False` in PyTorch).
- **Flexibility:** The solution allows the model to adapt with incoming data while preserving predetermined parameters.

- **Plot:** using dummy data we create this plot:
  <img width="926" alt="Screenshot 2025-03-19 at 5 35 02 PM" src="https://github.com/user-attachments/assets/853aa258-540e-455f-b86d-9b90bf3f76d1" />
  **What the plot shows:**
  - **X-Axis:** The number of iterations (or time steps) in the online learning process.
  - **Y-Axis:** The model’s loss value (or predicted probability), illustrating how the model’s performance changes over time.
  - **Trend:** The curve demonstrates the gradual reduction of loss as the model learns from the dummy data.
  - **Annotations:** In some parts of the plot, you might notice plateaus or changes in the curve’s slope, indicating moments when certain weights are fixed (i.e., not updated) during the learning process.
  
  This visualization helps confirm that even with fixed weights, the model is able to converge on a solution, and it highlights the impact of freezing parameters on the overall learning dynamics.

**Code/Detail Reference:**  
- [online_logistic_regression.py](./online_logistic_regression.py) *(Assuming the implementation details are in this file)*

---

## Question 4: Hedge Fund Trade Bug

**Short Summary:**  
Analyze a trading algorithm bug that reverses trades with a small probability (~0.1%). Evaluate the impact and propose the next steps.

**Key Points of the Solution:**  
- **Impact Quantification:** Approximately 10 trades per year are affected, with an estimated net loss of around $4,000 per year.
- **Risk Management:** Even a small performance drag should be eliminated to reduce risk and maintain trust in the system.
- **Actionable Step:** Prioritize debugging and fixing the execution code to ensure trades are executed as intended.
- **Importance:** Ensuring that no unintended flips occur is crucial for maintaining the model's integrity and investor confidence.

**Code/Detail Reference:**  

---

