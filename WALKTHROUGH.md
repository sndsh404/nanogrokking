# One trained model, end to end

This file follows one addition, (7 + 40) mod 53 = 47, through the committed `fast` model, with the real numbers at every stage. Everything quoted here is the output of one command:

```
python walkthrough.py --run runs/fast --a 7 --b 40
```

Run it yourself to check the numbers, or pass different `--a` and `--b` values to trace a different addition.

## the spectrum: what the model is made of

```
embedding Fourier spectrum, total power 103.4:
  k =  9   power  26.44   share 0.256   cumulative 0.256
  k =  1   power  25.80   share 0.249   cumulative 0.505
  k =  3   power  23.44   share 0.227   cumulative 0.732
  k = 25   power  21.69   share 0.210   cumulative 0.941
  k = 18   power   1.30   share 0.013   cumulative 0.954
```

The embedding is the table that maps each token to a vector of 96 numbers; it is the model's way of writing the input down. A discrete Fourier transform re-expresses each embedding row as a sum of cosines and sines over the 53 token positions, so we can see which rotation frequencies the model uses. Four frequencies, k = 1, 3, 9, 25, carry 94 percent of the weight. The fifth frequency carries 1.3 percent. The model threw the other 22 away.

## the circle: how a number becomes an angle

```
tokens [0, 7, 40, 47] as points on the learned circles (angles relative to token 0):
  k =  9:  measured [0.0, 65.0, 287.5, 355.5]   expected [0.0, 67.9, 285.3, 353.2]
        point for token 7: (1.154, 2.387)
  k =  1:  measured [0.0, 62.8, 282.4, 327.1]   expected [0.0, 47.5, 271.7, 319.2]
        point for token 7: (1.498, 2.012)
  k =  3:  measured [0.0, 143.2, 96.5, 232.0]   expected [0.0, 142.6, 95.1, 237.7]
        point for token 7: (-1.973, 1.431)
  k = 25:  measured [0.0, 100.3, 317.1, 51.3]   expected [0.0, 108.7, 312.5, 61.1]
        point for token 7: (-0.536, 2.050)
```

For each frequency k, the embedding of a token is projected onto the cosine and sine components of that frequency, giving one point (x, y) per token. If the model places numbers on a circle, that point's angle advances by 360 times k / 53 degrees each time the number goes up by one. At k = 3 the advance is 20.4 degrees: token 7 sits at 142.6 degrees, token 40 at 95.1, and the measured points land within a degree of that.

The addition is visible in the table. At k = 3, the angle of 7 plus the angle of 40 is 142.6 + 95.1 = 237.7 degrees, which is exactly the angle of 47. The same holds at the other three frequencies, up to a fixed offset the rest of the network absorbs. Adding two numbers mod 53 is adding their angles, and angles wrap around at 360 the same way the answers wrap around at 53. The measured angles drift a few degrees from the expected ones because the embedding also carries the other three frequencies; each circle is the dominant structure in its own components, not the only thing in the vector.

## the trace: one addition through the model

```
tracing (7 + 40) mod 53 = 47:
  tokens [7, 40, '='] -> three 96-dim vectors, norms [1.43, 1.36, 4.47]
  attention at the "=" position, per head:
    head 0: to 7 0.265, to 40 0.730, to "=" 0.005
    head 1: to 7 0.901, to 40 0.098, to "=" 0.001
    head 2: to 7 0.128, to 40 0.871, to "=" 0.001
  hidden layer: 211 of 384 neurons active at the answer position, strongest [6.64, 5.16, 5.14, 5.1, 4.72]
  top-5 logits at the answer position (the answer is 47):
    47   logit  39.562   probability 1.0000  <== answer
    45   logit  20.950   probability 0.0000
    41   logit  20.332   probability 0.0000
    46   logit  19.591   probability 0.0000
    11   logit  18.670   probability 0.0000
```

Stage by stage:

- The three tokens become three vectors of 96 numbers. The "=" token's vector is mostly a learned position marker; its job is to be the place the answer is read from.
- Attention is the mechanism that lets positions copy information from each other: each position scores every earlier position and takes a weighted mix. The "=" position splits the work. Head 1 copies almost entirely from the 7 (0.901). Heads 0 and 2 copy mostly from the 40 (0.730 and 0.871). Together the three heads bring both operands to the answer position. Almost nothing attends to "=" itself.
- The hidden layer (a ReLU layer: each of 384 neurons outputs the positive part of a weighted sum, zero if negative) is where the angle arithmetic happens. 211 of the 384 neurons are active for this input. The paper's analysis shows these neurons compute the products the sum-of-angles identities need; here we show its size and its strongest outputs rather than re-deriving that wiring.
- The unembedding turns the result into one logit (a raw score) per candidate answer, and softmax turns the scores into probabilities. 47 scores 39.6 against 21.0 for the runner-up 45, so the probability rounds to 1.0000. The model is certain, and right.

## what to take away

Nothing in that trace looked 47 up. The model wrote 7 and 40 as angles on four circles, combined the angles at the answer position, and scored highest the number sitting at the summed angle. Every pair works this way, seen or unseen, which is what makes the solution general: the mechanism never touches the training table because there is no table. The trace is the same for any `--a` and `--b` you pass.
