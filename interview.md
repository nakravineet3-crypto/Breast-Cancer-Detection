# Interview Prep — Every Question, Every Answer

> Read this as if you're rehearsing spoken answers, not writing an essay. Answers are written in
> first person, the way you'd actually say them. Practice saying them out loud, not just reading.

---

# Section A — The Elevator Pitch (get this rock-solid first)

**"Walk me through your project."**
> "I reproduced and critically evaluated a research paper on breast cancer detection from
> histopathology images, using a spiking neural network — a biologically-inspired architecture
> where neurons communicate via discrete pulses over simulated time, instead of continuous
> activations like a normal network. I built it incrementally: a trainable image filter, then
> spiking convolutional layers, then attention mechanisms and a recurrent layer, then a bio-inspired
> metaheuristic optimizer to refine the final layer. At every stage I measured the real effect on an
> identical, held-out test set — so I ended up with an honest, monotonic progression from 52%
> accuracy up to 83%, with every component's actual contribution documented, not just a final
> number. I also found and fixed two genuine bugs in the optimization algorithm along the way,
> which I can walk through in detail — they're actually the most interesting part."

**"What was the single most interesting technical problem you solved?"**
> "Probably the Shrike Optimization Algorithm bug. I translated the paper's equations into code,
> tested it on a toy problem with a known answer first, and found the algorithm was mathematically
> degenerate — one of its update rules compared a candidate to itself, which becomes a no-op the
> moment they match, which happens almost immediately. I instrumented the code to count how often
> each update rule actually won, confirmed it was winning zero times out of 750 attempts, and fixed
> it by comparing against a different, randomly-chosen candidate instead. Then when I ran it for
> real, the model's accuracy collapsed from 74% to under 20% — I dug in and found a second, totally
> different bug: the small subset of images I was using to evaluate fitness during search happened
> to be 100% one class, because the dataset file wasn't shuffled. The algorithm found a shortcut —
> always guess that one class — that looked perfect on the biased subset and was useless on real
> data. Fixed the sampling, and got a genuine, reproducible improvement: 74% to 83% accuracy."

---

# Section B — Tech Stack: Why These Choices

**"Why Python?"**
> "It's the standard for ML/DL work — every library I needed (PyTorch, snnTorch, scikit-learn)
> has first-class Python support, and the ecosystem for data handling (pandas, NumPy) is
> unmatched. There's no real alternative for a project like this."

**"Why PyTorch over TensorFlow?"**
> "Two reasons. First, PyTorch's imperative, define-by-run execution model made debugging much
> easier — I could literally print tensor shapes and values mid-forward-pass while diagnosing bugs,
> which was essential when I was tracking down issues like the spiking network's dead-initialization
> problem. Second, and more decisively: the spiking neural network library I needed, snnTorch, is
> built directly on top of PyTorch — that made the choice for me."

**"Why snnTorch specifically, and not build spiking neurons from scratch?"**
> "Spiking neurons need two non-trivial pieces to actually work with backpropagation: the
> Leaky Integrate-and-Fire dynamics themselves, and surrogate gradients — a workaround for the
> fact that a spike's derivative is zero or undefined almost everywhere, which would otherwise
> block gradient flow entirely. snnTorch implements both correctly and is well-tested. I did
> validate it wasn't a black box to me, though — I hand-traced a LIF neuron's behavior over several
> time steps by hand, then ran the exact same numbers through snnTorch and confirmed they matched
> exactly, before trusting it in the real model."

**"Why did you drop image resolution to 64×64? Wouldn't that hurt accuracy?"**
> "Yes, honestly — it's very likely the single biggest ceiling on my results. I did it because a
> spiking network's forward pass costs roughly 25× more than a normal CNN's, since it repeats the
> same computation once per simulated time step, and I was iterating through architecture changes
> weekly on a single consumer GPU. It was a deliberate speed-for-iteration-speed tradeoff during
> development. If I were optimizing purely for final accuracy rather than development velocity,
> raising resolution back up — even to 128×128 — would likely be the highest-leverage single change
> I could make."

**"Why NumPy for the optimizer instead of keeping everything in PyTorch tensors?"**
> "The Shrike Optimization Algorithm doesn't need gradients — it's a gradient-free search — so
> there's no benefit to PyTorch's autograd machinery there, and NumPy is simpler and has fewer
> moving parts for pure numerical array operations. I only convert into PyTorch tensors at the
> boundary, when a candidate solution needs to actually be loaded into the model to evaluate its
> fitness."

---

# Section C — Architecture Deep Dive

**"Explain your full architecture, layer by layer."**
> "Input image goes through a trainable image filter first — a learned edge-preserving smoothing
> filter, similar to a classical guided image filter, with one learnable parameter controlling how
> aggressively it smooths. Then three convolutional blocks, but instead of ReLU activations, each
> one uses a spiking neuron — Leaky Integrate-and-Fire — run for 25 simulated time steps. After the
> third conv block, I apply channel and spatial attention — CBAM — so the network learns which
> detected patterns and which image regions matter most. That feeds into a spiking recurrent layer,
> which has a learnable feedback connection from its own previous output, giving it short-term
> memory across time steps. Then a final fully-connected spiking layer produces two output scores —
> benign and malignant — at every time step, and I combine those 25 steps' outputs using a learned,
> softmax-normalized weight per time step, instead of just averaging them uniformly."

**"Why attention specifically — what does it buy you that a plain CNN doesn't have?"**
> "A plain CNN treats every detected feature and every spatial location with equal weight when it
> passes information to the next layer. Attention lets the network learn that some feature channels
> or some image regions are more informative than others for this specific input, and rescale them
> accordingly before they propagate further. It's the same underlying idea that made Transformers
> so effective in NLP and vision — I used a convolutional variant (CBAM) rather than full
> self-attention, since my inputs are spatial feature maps, not sequences."

**"Why put the recurrent layer where you did, after attention?"**
> "That follows the architecture diagram in the paper I was reproducing — spiking conv layers,
> then attention, then a spiking recurrent layer, then the output layer. Structurally it makes
> sense too: attention refines what the deepest convolutional features are telling you, and the
> recurrent layer then integrates that refined signal across the temporal dimension before making
> a final decision."

**"What's the difference between your spiking recurrent layer and a normal RNN/LSTM?"**
> "Conceptually similar — both maintain state across a sequence and use a feedback connection.
> The concrete difference is that my recurrent layer's neurons are still spiking — LIF dynamics,
> discrete 0/1 outputs, membrane potential — rather than continuous-valued LSTM/GRU cells. It's
> `snn.RLeaky` from snnTorch: an LIF neuron whose own previous spike output feeds back in as
> extra input each time step, with that feedback weight itself learned during training."

---

# Section D — Deep Learning Fundamentals (will come up regardless of your project)

**"What is backpropagation?"**
> "It's how you compute the gradient of the loss with respect to every parameter in the network,
> using the chain rule from calculus, propagated backward from the output layer to the input.
> Each layer only needs to know its own local derivative and the gradient signal coming from the
> layer after it — that's what makes it efficient enough to actually train deep networks."

**"What's the difference between SGD and Adam?"**
> "Plain SGD takes a step in the negative-gradient direction, scaled by one fixed learning rate for
> every parameter. Adam adapts the effective step size per-parameter, using running averages of
> both the gradient itself (momentum) and the squared gradient (an adaptive per-parameter scale) —
> so parameters that have consistently large or noisy gradients get smaller effective steps, and
> vice versa. I used Adam throughout because it's generally more forgiving of learning-rate choice
> and converges faster on most problems, including mine."

**"What is overfitting, and how did you guard against it?"**
> "Overfitting is when a model starts memorizing quirks of the training data rather than learning
> patterns that generalize — you see it as training accuracy climbing while validation accuracy
> stalls or drops. I used dropout in the fully-connected layers, monitored the train/val gap every
> epoch, and used early stopping — halting training once validation performance stopped improving
> for a set number of epochs, then restoring the best checkpoint rather than the final one."

**"What is data leakage, and where could it have happened in your project?"**
> "Data leakage is when information from outside the training set inadvertently influences the
> model — most commonly, when the test set isn't truly independent. In my case, the real risk was
> patient-level leakage: the dataset has multiple images per patient, and if the same patient's
> tissue appeared in both training and test, the model could partly recognize that specific tissue
> sample rather than learn general benign/malignant patterns, inflating my reported accuracy. I
> explicitly split by patient ID, not by image, and wrote a verification check confirming zero
> patient overlap across splits before trusting any results."

**"Explain the bias-variance tradeoff."**
> "High bias means the model is too simple to capture the real pattern — it underfits, performing
> poorly on both train and test data. High variance means it's too sensitive to the specific
> training data — it overfits, doing well on train but poorly on unseen data. You're always
> balancing model capacity against this tradeoff; things like regularization, dropout, and early
> stopping are all tools for pulling a model back from the high-variance end."

**"What is class imbalance, and how did you handle it?"**
> "It's when one class has substantially more examples than another, which biases a naively-trained
> model toward the majority class since predicting it more often lowers average loss even without
> real learning. I actually hit this directly — my first training run got 78% accuracy that looked
> good until I noticed it was only catching 22% of true benign cases, because the model had just
> learned to guess 'malignant' most of the time, since malignant outnumbers benign roughly 1.8:1 in
> my training data. I fixed it with a class-weighted loss function, so a mistake on the minority
> class counts proportionally more during training."

**"What's the difference between precision and recall, and when do you care more about one?"**
> "Precision is: of everything you predicted positive, how much was actually positive. Recall is:
> of everything actually positive, how much did you catch. In a medical context like mine, recall
> on the malignant class is usually the more important one to protect — missing a real malignant
> case (a false negative) is generally more costly than a false alarm on a benign case. That's
> actually relevant to a real tradeoff I found: my architecture improvements consistently increased
> malignant recall a lot (53%→94%) while benign recall dropped (50%→34%) — which I'd argue is the
> right direction to trade in a screening context, though it's worth stating explicitly rather than
> just reporting the accuracy number."

**"What is a confusion matrix?"**
> "A table of predicted label vs. true label counts — for binary classification, a 2×2 grid: true
> positives, false positives, true negatives, false negatives. Every other classification metric
> (precision, recall, F1, accuracy) is just a different summary computed from those four numbers."

**"What is batch normalization and why does it help?"**
> "It re-centers and rescales a layer's activations using the statistics of the current batch,
> after every layer. It helps because as data flows through many layers, activation magnitudes can
> drift to extreme values, making training unstable or slow. Renormalizing keeps things in a
> consistent, well-behaved range and generally lets you use higher learning rates safely."

**"What's the vanishing gradient problem, and did you encounter anything like it?"**
> "It's when gradients shrink to near-zero as they propagate backward through many layers, so
> earlier layers barely get updated. I encountered a closely related, spiking-specific version of
> this: in my LIF neurons, using the library's default firing threshold, I found that current
> magnitude shrank with network depth — deeper layers were firing on well under 1% of
> opportunities, essentially a 'vanishing spikes' problem. I diagnosed it by directly measuring
> per-layer firing rates, then fixed it by empirically tuning the threshold down until every layer
> stayed meaningfully active without saturating."

---

# Section E — Spiking Neural Networks (be ready for genuine curiosity here — it's niche)

**"What is a spiking neural network, and why would anyone use one over a normal network?"**
> "A spiking neural network's neurons communicate via discrete, binary pulses over simulated time,
> rather than continuous numbers computed instantaneously. Each neuron accumulates incoming signal
> into a 'membrane potential' that leaks over time, and only fires when that potential crosses a
> threshold. The main motivation is energy efficiency on specialized neuromorphic hardware — since
> neurons are silent most of the time and only compute when they actually spike, you get sparse,
> event-driven computation instead of every neuron computing something on every input, which is
> how standard ANNs work. It's a genuinely active research area, not yet mainstream in production
> the way standard deep learning is."

**"How do you train something with binary, non-differentiable outputs?"**
> "The core trick is a surrogate gradient. A spike is literally a step function — its true
> derivative is zero everywhere except an undefined jump at the threshold, which would completely
> block gradient-based learning. So during the backward pass only, you substitute a smooth
> approximation of that step function to get a usable gradient signal, while the forward pass still
> produces genuine discrete spikes. The forward and backward passes are technically inconsistent
> with each other, but it works well empirically and is the standard approach."

**"What's rate encoding, and did you use it?"**
> "Rate encoding means representing information by how frequently a neuron fires over a time
> window, rather than by a single instantaneous value. In my case, I didn't explicitly convert
> static images into spike trains via random sampling — instead, I fed the same preprocessed image
> to the network at every one of 25 simulated time steps, and let each LIF neuron's own
> accumulate/leak/fire dynamics naturally produce a time-varying spike pattern from that constant
> input. That's a simpler, commonly-used alternative to explicit rate encoding for static images."

**"What real problems did you run into that are specific to spiking networks, not regular ones?"**
> "Two big ones. First, threshold sensitivity — the default firing threshold left deeper layers
> almost completely silent, which I found by directly measuring firing rates layer by layer and
> fixed by tuning the threshold. Second, and more interesting: I found that random weight
> initialization can produce a network that's completely 'born dead' — fires zero spikes anywhere,
> for any input — purely from an unlucky draw. I tested 8 different random seeds and got exactly
> one dead network. Critically, unlike a standard ReLU network, a dead spiking network can't train
> its way out of that state, because zero spikes means zero gradient signal. I built an explicit
> safeguard: create a model, run one real batch through it as a smoke test, and discard and retry
> with a new random seed if literally nothing fired."

---

# Section F — Attention Mechanisms

**"Explain attention in general terms — not specific to your project."**
> "Attention is a mechanism for letting a model learn to weight different parts of its input
> differently, rather than treating everything uniformly. It's computed as a set of learned
> weights, usually normalized (e.g. via softmax) so they're interpretable as relative importance,
> which are then used to rescale or combine the underlying features. It's the foundational idea
> behind Transformers, where self-attention lets every position in a sequence weigh every other
> position when building its representation."

**"How does what you built (CBAM) relate to Transformer-style self-attention?"**
> "Same underlying philosophy — learned importance weighting — but a different, older, and
> cheaper mechanism suited to convolutional feature maps rather than sequences. Channel attention
> asks 'which of my detected feature types matter', spatial attention asks 'which pixel locations
> matter' — both computed via small learned sub-networks and a sigmoid, then applied by
> element-wise multiplication. Full self-attention in a Transformer instead computes pairwise
> relationships between every pair of positions via query/key/value projections and a softmax —
> much more expressive, but quadratically more expensive, and designed for sequence-like data.
> I chose CBAM because it matches the architecture I was reproducing and fits convolutional feature
> maps naturally."

**"You mentioned 'temporal attention' — what's that, and is it a standard technique?"**
> "It's a custom addition I made for this specific setup, not a term of art I'm claiming is
> standard. My spiking network produces an output at every one of 25 simulated time steps, and the
> straightforward thing to do is just sum or average them. Instead, I added one learnable weight
> per time step, passed through a softmax so they sum to 1, so the network can learn to trust some
> simulated moments more than others when forming its final decision — the same underlying idea as
> channel/spatial attention, just applied along the time dimension instead."

---

# Section G — Optimization Algorithms

**"Explain gradient descent vs. what you called a 'metaheuristic' — what's actually different?"**
> "Gradient descent uses calculus — the derivative of the loss with respect to every parameter —
> to compute the exact locally-improving direction, then takes a step that way. It's extremely
> efficient because one backward pass gives you a direction for every parameter simultaneously. A
> metaheuristic like the one I implemented has no such shortcut — it maintains a population of
> complete candidate solutions and evolves them toward better ones purely through trial and error,
> guided only by a fitness score. That makes it far less efficient per-parameter, but it doesn't
> require the objective to be differentiable at all, which is sometimes useful for problems where
> gradients aren't available or the objective is discontinuous."

**"Why didn't you just use the metaheuristic to train the whole network?"**
> "Scale. Gradient descent can update tens of thousands of parameters in one backward pass because
> calculus gives you all their directions at once. A population-based search has no equivalent —
> every candidate is an independent, complete guess evaluated by brute-force trial, so the search
> space grows catastrophically with dimensionality. I made the deliberate, scoped decision to only
> search my final classifier layer — 66 numbers — with everything else frozen at its already
> gradient-trained value. That's actually consistent with how most real papers claiming
> 'metaheuristic-optimized deep learning' work under the hood, even when they don't say so
> explicitly."

**"What other metaheuristics are there, and how does yours relate to them?"**
> "The Shrike Optimization Algorithm I implemented is part of a broad family of nature/animal-
> behavior-inspired population-based search algorithms — Genetic Algorithms, Particle Swarm
> Optimization, Simulated Annealing, Cuckoo Search are other well-known examples. They generally
> share the same skeleton: maintain a population, evaluate fitness, apply some combination of
> exploitation (move toward known-good solutions) and exploration (random search to avoid getting
> stuck), repeat. Mine specifically uses a bird-feeding metaphor for its update rules — I found and
> fixed a real bug where one of those update rules degenerated into comparing a candidate against
> itself, which I fixed by making it compare against a different, randomly-selected candidate
> instead — structurally similar to the mutation operator in Differential Evolution, another
> well-known algorithm in this same family."

**"Walk me through a bug you found in your optimizer, in detail."**
> "I validated the algorithm on a toy problem first — a simple function with a known optimal answer
> — before trusting it on my real model. On that toy problem, it improved initially, then plateaued
> far short of the true optimum, and running 10× more iterations barely moved it. That told me it
> was stuck, not just slow. I instrumented the code to count which of the algorithm's three update
> rules actually won at each step, and found one of them — the 'exploit toward this candidate's own
> best-known position' rule — won zero times out of 750 attempts. The reason: the update formula
> was `step_size × (candidate − own_best) + own_best`. The moment a candidate matches its own
> tracked best — which happens almost immediately after any improvement — that expression collapses
> to just `own_best`, regardless of the step size. It's a mathematical no-op. I fixed it by
> comparing each candidate against a different, randomly-chosen member of the population instead,
> re-ran the toy test, and confirmed real, if imperfect, convergence was restored."

---

# Section H — Evaluation & Metrics

**"Your AUC was only 0.58 but accuracy was 83% — isn't that a red flag?"**
> "It looks like one at first glance, which is exactly why I dug into it rather than just reporting
> the number. AUC is a purely rank-based metric — it measures how well your confidence scores order
> true positives above true negatives across every possible threshold. My model's output is a spike
> count over only 25 discrete time steps, so its score space is inherently small — I found only 33
> unique probability values across all 1,282 test images. With that much tying between classes at
> shared score values, AUC gets heavily penalized even though threshold-based decisions — is the
> score above or below 0.5 — stay accurate. I confirmed this visually too: the ROC curve hugs the
> diagonal almost everywhere except one sharp jump at the one threshold that actually separates the
> classes well. My conclusion: AUC is structurally a poor fit for evaluating a discrete,
> spike-count-based output layer — it's a property of the architecture, not evidence of a weak
> classifier."

**"How confident are you in your final numbers, given a small test set?"**
> "Honestly, only moderately — and I'd say that directly in an interview. My test set is only 13
> patients, 1,282 images. I demonstrated directly, through repeated experiments, that this causes
> real evaluation noise — validation accuracy swinging significantly between architecturally-similar
> runs. The methodologically correct fix is patient-level k-fold cross-validation, training and
> evaluating across several different patient splits and averaging — I identified this as the right
> fix but didn't have time to implement it given the project timeline. I'd flag that explicitly
> rather than overstate confidence in a single number."

**"What would you do to make your evaluation more rigorous?"**
> "K-fold cross-validation at the patient level, as above. I'd also want a proper held-out
> validation set for threshold tuning, since right now my decision threshold is the default 0.5,
> not something tuned for the precision/recall tradeoff that actually matters clinically. And I'd
> want to report confidence intervals on my metrics, not just point estimates, given how few
> patients are in each split."

---

# Section I — Scaling & Production Questions

**"How would you scale this to a production system handling millions of images?"**
> "A few layers to that. First, training: I'd move off a single consumer GPU to a distributed
> multi-GPU setup with data parallelism — split batches across GPUs, sync gradients. Given the
> spiking network's 25× per-image cost from simulating time steps, I'd also look hard at reducing
> that — either fewer time steps if accuracy holds, or exploring whether the spiking layers can run
> more efficiently on actual neuromorphic hardware, which is the whole point of this architecture
> family. For inference at scale, I'd batch requests, consider quantizing the model, and since
> spiking networks are event-driven by nature, they're a genuinely good fit for specialized
> low-power inference hardware if that's available — though on standard GPU/CPU infrastructure,
> a regular CNN would likely be faster and simpler to deploy for the same accuracy."

**"How would you deploy this as a real service?"**
> "Wrap the trained model behind an inference API — load the weights once, serve predictions over
> HTTP or gRPC, batch incoming requests where latency budgets allow. I'd add input validation
> (image format/size checks), monitoring on prediction distribution over time (to catch drift — if
> the fraction of malignant predictions suddenly shifts, that's worth investigating), and I'd
> version the model artifact explicitly so I can roll back if a new version regresses."

**"What would you monitor in production, and why?"**
> "Model-level: prediction distribution drift, confidence score distribution, and if ground truth
> ever becomes available with a delay (e.g. confirmed diagnoses), actual accuracy over rolling
> windows. Data-level: input distribution drift — new hospitals or scanners could shift image
> characteristics in ways my training data doesn't cover. Infrastructure-level: latency, error
> rates, GPU utilization if serving on GPU. For a medical application specifically, I'd also want
> strong logging/auditability of every prediction given the stakes."

**"How would you handle a much larger dataset than 8,000 images?"**
> "The patient-wise splitting logic is already dataset-size-independent, so that part scales as-is.
> For training, I'd move to a proper data pipeline that streams from disk/cloud storage rather than
> loading everything into memory, use a distributed data loader, and likely move to multi-GPU
> training. I'd also revisit my image resolution decision — with more compute available, I'd want
> to go back up from 64×64 to preserve more diagnostic detail, since that was purely a
> development-speed tradeoff I made under my specific hardware constraints, not a
> fundamental limitation of the approach."

**"How would you reduce training time?"**
> "Several independent levers: mixed-precision training (fp16) to speed up GPU compute, reducing
> the number of simulated time steps if accuracy tolerates it (this directly and linearly reduces
> the spiking network's per-image cost), increasing batch size to better utilize the GPU, and
> caching preprocessed images to disk instead of re-decoding and resizing from raw files every
> epoch, which I currently do on the fly."

---

# Section J — Pros, Cons, and Critical Thinking

**"What are the actual pros and cons of using a spiking neural network here, honestly?"**
> "Pros: genuine potential energy efficiency on the right hardware, and it's a legitimate,
> differentiated research angle rather than 'just another CNN paper.' Cons, and I'd be upfront
> about these: on standard GPU hardware — which is what almost everyone actually has — a spiking
> network is strictly slower than an equivalent CNN, since it repeats computation across time
> steps for no inference-time benefit unless you have neuromorphic hardware to actually run it on.
> Training is also empirically harder — slower gradient signal buildup, threshold sensitivity,
> and the dead-initialization problem I found are all real, extra engineering burdens a standard
> CNN doesn't have. For this specific project, a well-tuned standard CNN might well match or beat
> my spiking network's accuracy with far less engineering effort — the point of doing it this way
> was to explore and validate the architecture family, not to claim it's currently the practically
> superior choice on commodity hardware."

**"If accuracy were the only goal, would you still choose this architecture?"**
> "Honestly, probably not from scratch — I'd likely start with a well-established CNN or vision
> transformer backbone with transfer learning from a large pretrained model, which would likely
> reach higher accuracy with far less training instability and engineering overhead than what I
> built. I chose this architecture because I was specifically reproducing and critically evaluating
> a paper that used it, which was the actual goal of the project — understanding a less-common
> architecture family deeply, not maximizing accuracy by any means available."

**"What's the biggest weakness of your current results?"**
> "The small, noisy test set, honestly — 13 patients isn't enough to fully trust a single accuracy
> number, and I demonstrated that noise directly during development. If I had to pick one thing to
> fix before treating these results as final, it's proper cross-validation, not further
> architecture tweaking."

**"What would you change about your own approach, in hindsight?"**
> "I'd add the fitness-subset class-balance check I eventually built for the optimizer much
> earlier, as a standard practice — the second SOA bug (100%-one-class subset) was avoidable if
> I'd verified data balance by default before trusting any subset, rather than after a run
> collapsed. That's a general lesson: verify assumptions about data explicitly, don't assume a
> slice of a dataset is representative just because it's the right size."

---

# Section K — Behavioral / STAR-Format Answers

**"Tell me about a time you found a bug that wasn't obvious."**
> **Situation**: My optimization algorithm's first real run on the actual model collapsed test
> accuracy from 74% to under 20%.
> **Task**: Figure out why, rather than just accept a bad result or blindly retry.
> **Action**: I checked the confusion matrix first and noticed the model was predicting one class
> for almost everything — 100% recall on benign, near-zero on malignant. That was specific enough
> to investigate: I checked the actual data being used to evaluate fitness during the search, and
> found the subset was pulled as the first 128 rows of an unshuffled file — which turned out to be
> 100% benign, because the dataset was grouped by patient/scan order, not randomized.
> **Result**: Fixed the sampling to be properly randomized and verified balanced before trusting it
> again, and also fixed a related root problem (accuracy as a fitness metric can saturate and lose
> discriminating power) by switching to a continuous loss-based fitness signal. Final result: a
> genuine, reproducible 74%→83% improvement.

**"Tell me about a technical decision you made under a real constraint."**
> **Situation**: A spiking network's forward pass costs about 25× more per image than a normal
> CNN, since it repeats computation across simulated time steps, and I only had a single 4GB
> consumer GPU.
> **Task**: Keep architecture iteration fast enough to actually make weekly progress.
> **Action**: I deliberately dropped image resolution from what a full CNN could handle (128px) down
> to 64px for every spiking model, benchmarking actual epoch times before committing to a full
> training run each time rather than guessing.
> **Result**: Made rapid iteration genuinely possible — full training runs completed in minutes, not
> hours — at the honest cost of image detail, which I can clearly name as the most likely reason my
> final accuracy sits well below the original paper's. I'd reverse this decision first if given more
> compute.

**"Tell me about a time you had to explain a technical limitation honestly rather than hide it."**
> **Situation**: My model's AUC score (0.58) looked bad next to its accuracy (83%) — an easy metric
> to quietly omit or gloss over.
> **Task**: Decide how to represent that in my results.
> **Action**: I investigated rather than hid it — traced it to the discrete, quantized nature of a
> spike-count output producing only 33 unique scores across the whole test set, and confirmed it
> visually via the ROC curve's shape.
> **Result**: Turned what could have looked like a weakness into a documented, defensible finding
> about a structural property of spike-count-based classifiers — and I can explain exactly why the
> number looks the way it does, rather than hoping nobody asks.

---

# Section L — Curveball Questions

**"If I gave you one more month, what's the highest-leverage thing you'd do?"**
> "Raise image resolution back up and extend the training budget — those are the two factors most
> likely responsible for the gap between my results and the original paper's, and both are direct,
> well-understood fixes rather than open research questions."

**"What don't you know about your own project that you wish you did?"**
> "Whether my results would hold up under proper cross-validation — I never got to implement it,
> so I genuinely don't know if 83% is a robust number or partly an artifact of one particular
> 13-patient test split. I'd say that directly rather than overclaim confidence."

**"Could this model be biased in a way you haven't checked for?"**
> "Very possibly — I haven't checked whether performance varies systematically across the four
> magnification levels, across tumor subtypes, or across individual patients in ways that could
> indicate the model is picking up on something other than the actual diagnostic signal. That's a
> real gap I'd want to close before treating this as production-ready in any sense."

**"Why should I trust your 83% number given everything you just told me about its limitations?"**
> "You shouldn't trust it blindly — and that's exactly the point. I'd rather hand you a number with
> its actual limitations clearly stated than a clean-looking number I can't defend under
> questioning. What I can stand behind is the methodology: every comparison in this project was
> measured on an identical test set, every suspicious result was investigated rather than accepted,
> and every architectural change's real effect is documented, including the ones that made things
> worse in some dimension. That process is what I'd want you to trust, more than any single digit."
