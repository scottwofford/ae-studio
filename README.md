# ae-studio
take-home test for data scientists

### Data Science Questions
This repository contains answers and code snippets for several data science questions from AE studio. Each question is addressed with a succint summary, key solution points, and links to the code.

### Time Spent and Tools Used
Time spent: in between 45 min and 1 hour 

Tools used: ChatGPT and Claude (via Cursor) 

---

## Question 1: Appending "!" to a List of Strings

**Summary:**  
Create a new list by appending "!" to each string in a list. 

**Solution:**  
- **List Comprehension:** A concise one-line method that iterates over each string and appends "!".
- **Recursion:** A function that processes the list element by element, appending "!" and recursively handling the remainder of the list.

**Code:**  
**List Comprehension:**
```
def append_exclamation_list_comp(strings):
    """Return a new list with '!' appended to each string using list comprehension."""
    return [s + "!" for s in strings]
```
**Recursion:**
```
def append_exclamation_recursive(strings):
    """Return a new list with '!' appended to each string using recursion."""
    if not strings:  # base case: if the list is empty, return an empty list
        return []
    # recursive case: append "!" to the first string and recurse on the remainder of the list
    return [strings[0] + "!"] + append_exclamation_recursive(strings[1:])
```
---

## Question 2: Email Priority Classification

**Summary:**  
Determine the importance of the `is_contact` feature when classifying email priority. The feature indicates whether the sender is in the recipient’s contact list.

**Solution:**  

- **Clarification on Data Timing:** In a first conversation with the client, I would ask something like:
  
  “Does is_contact indicate the sender was in the recipient’s address book at the time the email was received, or was it updated later (e.g. if the recipient added the sender to contacts afterward)?”
  
  - If `is_contact` is determined at email receipt time, then it’s a valid feature for modeling (it’s information that would genuinely be available when predicting priority for new incoming emails).
  - If instead the contact list is updated over time and the dataset simply marks whether the sender is in the contact list now or at the end of the two-year period, then some emails could be marked as `is_contact` = `True` even though at the time they were received, the sender was not yet in the contact list. This would mean the model is using information that would not have been available when the email arrived. In other words, it would be leaking future knowledge (the fact that the sender was eventually added to contacts) into the training data. This could artificially inflate model performance and would not generalize to real future emails.
- **Implications:** This clarification determines if the feature can be reliably used for training and prediction.

---

## Question 3: Online Logistic Regression with Fixed Weights

**Summary:**  
Outline an approach for implementing an online logistic regression model where some weights remain fixed during learning.

**Solution:**  
- **Model Initialization:** Define weights and identify which ones are fixed.
- **Online Updates:** Process each data point sequentially using stochastic gradient descent, updating only the trainable weights.
- **Library Support:** Options include customizing optimizers in libraries like PyTorch or TensorFlow by freezing fixed weights via flags (e.g., `requires_grad=False` in PyTorch).
- **Flexibility:** The solution allows the model to adapt with incoming data while preserving predetermined parameters.

- **Plot:** using dummy data we create this plot:
  <img width="926" alt="Screenshot 2025-03-19 at 5 35 02 PM" src="https://github.com/user-attachments/assets/853aa258-540e-455f-b86d-9b90bf3f76d1" />
  **What the plot shows:**
  - **X-Axis:** The number of iterations (or time steps) in the online learning process.
  - **Y-Axis:** The model’s loss value (or predicted probability), illustrating how the model’s performance changes over time.

**Code/Detail Reference:**  
- [online_logistic_regression.py](./online_logistic_regression.py)

---

## Question 4: Hedge Fund Trade Bug

**Summary:**  
Analyze a trading algorithm bug that reverses trades with a small probability (~0.1%). Evaluate the impact and propose the next steps.

**Solution:**  
- **Impact Quantification:** Approximately 10 trades per year are affected, with an estimated net loss of around $4,000 per year. See below for detail. 
- **Rationale for fixing:** Even though impact to profits is small, Fixing the bug eliminates an avoidable error, boosting reliability and consistency so that trade performance truly reflects the model and market, not sporadic glitches.
- **Cost of fix/Cap on bugfixing effort:** Given a $400k salary (about $200/hour) and a $4k annual loss from the bug, it makes sense to spend around 20 hours fixing the issue (or slightly more for benefits from a better system).
- **Action:** Check with the team to scope the effort and work with managers/leaders to determine the relative importance of fixing the bug compared to other items in the backlog.

**Impact Estimation Detail:**  
- If there were no bugs, the model’s edge of 52% win rate would yield an expected profit. For each trade, the expected value is 0.52*(+$5,000) + 0.48*(-$5,000) = $200 per trade on average. Over 10,000 trades, that’s an expected +$2,000,000.
- With the bug, in those ~10 affected trades, about 5 or so will turn a win into a loss (losing what would have been a +$5k gain and incurring a -$5k loss instead, effectively $10k less than expected for each of those trades) and ~5 will turn a loss into a win (each $10k more than expected). The net effect is slightly negative because the model wins slightly more often than it loses. Roughly, we’d lose 5.2 * $10k due to flipped good trades and gain 4.8 * $10k from flipped bad trades. That comes out to a net loss of about 0.4 * $10,000 = $4,000 over the year relative to if there were no bugs. In terms of win rate, the effective win rate might drop from 52% to about 51.996% – a virtually negligible change.
- Besides the expected value, consider the risk/variability introduced: Each bug occurrence is a $10k swing. With ~10 such events, the standard deviation in bug impact could be a few $10k’s. This adds a bit of uncertainty to our yearly results (though compared to 10,000 trades, it’s still small noise).
---

