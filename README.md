# Fraud Detection - Using Self Organized Maps(Unsupervised Deep Learning)


### How do SOMs Learn?
- E.g. three features in our input vectors, nine nodes in output
- SOMs used to reduce dimensionality of the dataset, 
- Three features => Columns, thousands of thousands rows
- SOMs output is always 2D
- SOMs familiar to us ANN, RNN, CNN
   - Same network, only difference is the position of the nodes
   - SOMs are very different from supervised neural networks
      - 1) Much much easier than ANN, simple and straightforward
      - 2) Concepts that have the same name, but different meaning, might confuse meanings
   - Three synapases to the top output node (weight assigned, w1.1, w1.2, w1.3) 
   - Weights have a different connotation than ANN, in ANN weights were used to multiply the input by the weight added up and applied an activation function
   - No activation function in SOMs
   - Weights are a characteristics of the node itself, node has coordinates
   - Node like a ghost in our input space, where it can fit, weights are the coordinates of the node in the input space
   - Each is a ghost or an imaginary data point, trying to blend in
   - Has its own weights at the start of the algorithm
   - So why is this important, core of the self-organizing map algorithm
   - rows which of these nodes is closets to our rows in our dataset
   - competition -> every node which is the closest to our input state - euclidean distance, get values close to 1 (0 and 1 input, normalization or standardization) 
   - Row #1 to each node, BMU (Best Matching Unit) the closest -> core of the SOM algorithm
   - What happens next?
      - Found BMU, SOM will update 'weights' - weights are characteristic of nodes in SOM, even closer to our first row in our data set
      - SOM is coming closer to the datapoint, drag the SOM closer to the point
      - Self-organizes onto input data, some of the nearby points are being dragged closer to this point
      - COme closer to the row that they matched up to, closer to BMu heavier weights of data, weights are updated less
   - Whole chain, whole structure same direction, harder pulled BMU matched up with, radius concept works
   - How do they fight with each other? (Green BMU, Blue BMU, Red BMU) -> pulled much harder and becomes, greenish blue, pull on green and blue, both have an impact
### How do SOMs Learn Pt.2
   - Best Matching Units (BMUs) are updated
   - Sophisticated example (5 BMUs)
   - BMUs updated even closer to row they matched up to, area around it radius, radius selected at beginning is usually - nodes are updated
   - Purple node (row that is matched up), dragged closer and closer - resistance push and pull, new epoch, unique KOHONAN learning algorthm
   - radius starts shrinking through the algorithm - pulling on nodes, radius is shrinked, nodes are pulled - process becomes more and more accurate, more and more iterations (precise, more laser specific manner), mask for your data
   - battle between different nodes, settled into some kind of representation
   - Takeaways from the tutorial:
      - 1) SOMs retain topology of the input set - does everything it can to be as close to the data, like a mask for dataset, understanding datasets better
      - 2) SOMs reveal correlations that are not easily identified SOMs can neatly put all anyalze into a mso, see into map easily
      - 3) SOMs classify data without supervision - don't actually need labels (CNNs => train our dataset to recognize objects, after lots and lots of iterations - don't need any labels - SOMs will extract feature or show us dependencies and correlation not expecting) - used in scenarios you don't know what you're looking for
      - 4) No target vector -> no backpropagation - unsupervised, we don't actually have a target vector, no lateral connection between output nodes, pull on one node other gets pulled, radius we outline
      - 5) No Neural Network - output, there's a grid behind, are indeed on a self-organizing map
   - Soft math on SOMs => http://ai-junkie.com/ann/som/som1.html 
- Live Example of SOMs
   - Inputs of eight colors (Red, green, blue, orange, yellow, purple, RGB code for each color)
   - From 0 to 1, 8 rows, three columns into SOMs
   - Starting weights at random, dark blue (perserve topology) 
   - Very simple SOM in action

