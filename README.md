


In this project, we will be constructing and training a simple character-level recurrent neural network (RNN) for word classification. The purpose of these tutorials is to demonstrate the process of preprocessing data for natural language processing (NLP) modeling from scratch, without relying on the convenience functions provided by torchtext. By doing so, we will gain a deeper understanding of the underlying principles of NLP preprocessing.

Unlike traditional word-level models, a character-level RNN interprets words as a sequence of individual characters. At each step, the model generates a prediction and a “hidden state,” which is then passed on to the subsequent step. The final prediction of the model determines the class or language to which the word belongs.

The specific task at hand involves training the model on several thousand surnames sourced from 18 different languages. The objective is to predict the language of origin based on the spelling of a given surname.

Let's start…

## Preparing the Data
In the data/names directory, there are 18 text files named after their respective languages, such as [Language].txt. These files contain numerous names, with each name appearing on a separate line. Although most of the names are already in Romanized form, we still need to convert them from Unicode to ASCII.

To organize the data, we create a dictionary called category_lines where each category corresponds to a list of names. In our case, the categories represent languages. Additionally, we maintain two other variables: all_categorieswhich is a list containing all the language names, and n_categories which denotes the total number of categories/languages. These variables are stored for future reference and extensibility purposes.

## Turning Names into Tensors
Once the names are organized, we need to convert them into tensors to utilize them effectively. Each individual letter is represented using a “one-hot vector” of size <1 x n_letters>. In a one-hot vector, all elements are set to 0 except for the index corresponding to the current letter, which is set to 1. For instance, “b” would be represented as <0 1 0 0 0 ...>.

To construct a word, we concatenate these one-hot vectors to form a 2D matrix of size <line_length x 1 x n_letters>. The additional dimension of size 1 is included because PyTorch assumes that everything is in batches. In this case, we are using a batch size of 1, meaning we process one name at a time.

## Creating the Network
Before the introduction of autograd in Torch, creating a recurrent neural network (RNN) involved manually cloning layer parameters across multiple time steps. These layers held hidden state and gradients, which had to be managed separately. However, with autograd, the graph handles these aspects, allowing for a more streamlined implementation of the RNN using regular feed-forward layers. A LogSoftmax the layer is applied after the output layer to obtain probabilities for each language.

![Z2xbySO](https://github.com/AhmetTasdemir/Classifying_Names_With_a_Character_Level_RNN_PyTorch/assets/79527973/303ab860-f9d3-4cc4-a24e-6b9d1841cfba)


To run a step of this network we need to pass an input (in our case, the Tensor for the current letter) and a previous hidden state (which we initialize as zeros at first). We’ll get back the output (probability of each language) and a next hidden state (which we keep for the next step).

For the sake of efficiency we don’t want to be creating a new Tensor for every step, so we will use lineToTensor instead of letterToTensor and use slices. This could be further optimized by precomputing batches of Tensors.

## Training
### Preparing for Training
Before delving into the training phase, it is essential to create a few utility functions. One of these functions is responsible for interpreting the network’s output, which we know represents the likelihood of each category. To determine the predicted category, we can employ the topk the function of tensors, which returns the index of the maximum value.

### Training the Network
Now all it takes to train this network is to show it a bunch of examples, have it make guesses, and tell it if it’s wrong.

The loss function nn.NLLLoss is appropriate since the last layer of the RNN is nn.LogSoftmax.

Each loop of training will:

* Create input and target tensors
* Create a zeroed initial hidden state
* Read each letter in and
* Keep hidden state for next letter
* Compare the final output to the target
* Back-propagate
* Return the output and loss
Now we just have to run that with a bunch of examples. Since the train the function returns both the output and loss we can print its guesses and also keep track of loss for plotting. Since there are 1000s of examples we print only every print_every examples, and take an average of the loss.

### Plotting the Results
Plotting the historical loss from all_losses shows the network learning:


## Evaluating the Results
To see how well the network performs on different categories, we will create a confusion matrix, indicating for every actual language (rows) which language the network guesses (columns). To calculate the confusion matrix a bunch of samples are run through the network with evaluate(), which is the same as train() minus the backprop.

![sphx_glr_char_rnn_classification_tutorial_001](https://github.com/AhmetTasdemir/Classifying_Names_With_a_Character_Level_RNN_PyTorch/assets/79527973/5c9f5046-8d9d-4d88-9a06-d356a36a6eca)

You can pick out bright spots off the main axis that show which languages it guesses incorrectly, e.g. Chinese for Korean, and Spanish for Italian. It seems to do very well with Greek, and very poorly with English (perhaps because of overlap with other languages).
![sphx_glr_char_rnn_classification_tutorial_002](https://github.com/AhmetTasdemir/Classifying_Names_With_a_Character_Level_RNN_PyTorch/assets/79527973/21130b69-83cf-4d20-aec3-54656e4e3896)
