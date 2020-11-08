import numpy as np

training_spam = np.loadtxt(open("path"), delimiter=",")
print("Shape of the spam training data set:", training_spam.shape)
print(training_spam)

def estimate_log_class_priors(data):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column, calculate the logarithm of the empirical class priors,
    that is, the logarithm of the proportions of 0s and 1s:
    log(P(C=0)) and log(P(C=1))

    :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                 the first column contains the binary response (coded as 0s and 1s).

    :return log_class_priors: a numpy array of length two
    """

    dataSize = len(data)
    ham = 0 
    spam = 0
    
    #Amount of 0s and 1s
    for i in data:
        if i[0] == 0:
            ham += 1
        else:
            spam += 1
    
    #Calculate proportions 
    ham_prop = ham / dataSize
    spam_prop = spam / dataSize
    
    #Empirical class priors
    ham_priors = np.log(ham_prop)
    spam_priors = np.log(spam_prop)
    log_class_priors = np.array([ham_priors,spam_priors])
    
    return log_class_priors

##Check whether the returned objects of your function are of the right data type.
log_class_priors = estimate_log_class_priors(training_spam)
print("result", log_class_priors)

# Check length
assert(len(log_class_priors) == 2)

# Check whether the returned object is a numpy.ndarray
assert(isinstance(log_class_priors, np.ndarray))

# Check wehther the values of this numpy.array are floats.
assert(log_class_priors.dtype == float)

# Check wehther the values are both negative (the logarithm of a probability 0 < p < 1 should be negative).
assert(np.all(log_class_priors < 0))


##Estimate class-conditional likelihoods
def estimate_log_class_conditional_likelihoods(data, alpha=1.0):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column and binary features (words), calculate the empirical
    class-conditional likelihoods, that is,
    log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

    Assume a multinomial feature distribution and use Laplace smoothing
    if alpha > 0.

    :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

    :return theta:
        a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
        logarithm of the probability of feature i appearing in a sample belonging 
        to class j.
    """
    
    #Spam
    spamInd = (data[:,0] != 0)
    spamData = data[spamInd]
    # Total count of the keyword appears in the message
    sumSpam = spamData.sum(axis = 0)  
    spamS = np.delete((sumSpam),0)
    # Total count of keywords for all messages
    spamWords = np.sum(spamS)
    # k different words
    spamFeatures = len(spamS) 
    
    #Laplace-smoothing
    for i in range(spamFeatures):
        spamS[i] = (spamS[i] + alpha) / (spamWords + (spamFeatures * alpha))
    
    spamS = np.log(spamS)
    
    #Ham 
    hamInd = (data[:,0] == 0)
    hamData = data[hamInd]
    # Total count of the keyword appears in the message
    sumHam = hamData.sum(axis = 0)
    hamS = np.delete((sumHam),0)
    # Total count of keywords for all messages
    hamWords = np.sum(hamS)
    hamFeatures = len(hamS)
    
    #Laplace-smoothing
    for i in range(hamFeatures):
        hamS[i] = (hamS[i] + alpha) / (hamWords + (hamFeatures * alpha))
    
    hamS = np.log(hamS)
    theta = np.array([hamS, spamS])
    
    return theta

##check whether the returned objects of your function are of the right data type.
log_class_conditional_likelihoods = estimate_log_class_conditional_likelihoods(training_spam, alpha=1.0)
print(log_class_conditional_likelihoods)

# Check data type(s)
assert(isinstance(log_class_conditional_likelihoods, np.ndarray))

# Check shape of numpy array
assert(log_class_conditional_likelihoods.shape == (2, 54))

# Check data type of array elements
assert(log_class_conditional_likelihoods.dtype == float)

##Classify e-mails
def predict(new_data, log_class_priors, log_class_conditional_likelihoods):
    """
    Given a new data set with binary features, predict the corresponding
    response for each instance (row) of the new_data set.

    :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
    :param log_class_priors: a numpy array of length 2.
    :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
        theta[j, i] corresponds to the logarithm of the probability of feature i appearing
        in a sample belonging to class j.
    :return class_predictions: a numpy array containing the class predictions for each row
        of new_data.
    """
    # Initialize class predictions array
    class_predictions = []
    
    #Loop through email and predict
    for i in new_data:
        spamProb = log_class_priors[1] + (np.sum(log_class_conditional_likelihoods[0] * i))
        hamProb = log_class_priors[0] + (np.sum(log_class_conditional_likelihoods[1] * i))
        classProb = np.array([hamProb, spamProb])
        predClass = np.argmax(classProb)
        class_predictions.append(predClass)
    
    class_predictions = np.array(class_predictions)
    
    return class_predictions

def accuracy(y_predictions, y_true):
    """
    Calculate the accuracy.
    
    :param y_predictions: a one-dimensional numpy array of predicted classes (0s and 1s).
    :param y_true: a one-dimensional numpy array of true classes (0s and 1s).
    
    :return acc: a float between 0 and 1 
    """
    totalMails = len(y_predictions)
    accuracy = 0
    for i in range(totalMails):
        if y_predictions[i] == y_true[i]:
            accuracy += 1
    
    acc = accuracy / totalMails
    print(acc)
    
    return acc

##check whether the returned objects of your function are of the right data type.
class_predictions = predict(training_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)

# Check data type(s)
assert(isinstance(class_predictions, np.ndarray))

# Check shape of numpy array
assert(class_predictions.shape == (1000,))

# Check data type of array elements
assert(np.all(np.logical_or(class_predictions == 0, class_predictions == 1)))
       
# Check accuracy function
true_classes = training_spam[:, 0]
training_set_accuracy = accuracy(class_predictions, true_classes)
assert(isinstance(training_set_accuracy, float))
assert(0 <= training_set_accuracy <= 1)


##Classifying previously unseen data
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")
print("Shape of the testing spam data set:", testing_spam.shape)
testing_spam

##Classify all messages in the testing_spam data set
log_class_priors = estimate_log_class_priors(testing_spam)
print("result", log_class_priors)
print("\n")

log_class_conditional_likelihoods = estimate_log_class_conditional_likelihoods(testing_spam, alpha=1.0)
print(log_class_conditional_likelihoods)
print("\n")

true_classes = testing_spam[:, 0]
class_predictions = predict(testing_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)
testing_set_accuracy = accuracy(class_predictions, true_classes)

