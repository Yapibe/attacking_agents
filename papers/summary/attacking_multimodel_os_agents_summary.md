# Summary of "Attacking Multimodal OS Agents"

## Abstract

The paper introduces a novel security vulnerability in multimodal OS agents - AI systems that can interact with computer interfaces through vision-language models. These agents use APIs to control mouse movements, keyboard inputs, and capture screenshots to complete tasks autonomously.

Key findings:
- The authors identify a new attack vector called "malicious image patches" (MIPs)
- These patches are adversarially designed images that can be embedded in desktop backgrounds or shared on social media
- When these images appear in screenshots captured by OS agents, they can trick the agent into performing harmful actions (like visiting malicious websites)
- The attacks are generalizable across different user requests, screen layouts, and multiple OS agent implementations
- This represents a significant security concern that needs addressing before widespread adoption of OS agents

## Introduction

The introduction establishes the context and significance of the research on OS agent vulnerabilities:

### Evolution of AI Systems
- Large language models (LLMs) and vision-language models (VLMs) have made significant advancements
- These models are typically fine-tuned to be "helpful and harmless"
- Previous research has shown adversarial attacks can bypass these safeguards

### Paradigm Shift: From Passive to Active Agents
- Traditional LLMs/VLMs generate content that isn't directly actionable
- OS agents represent a fundamental shift - they interact with computer systems through APIs
- These agents perform actions like mouse clicks, keyboard inputs, and screenshot captures
- This transforms AI from passive information sources to active participants that can impact digital and physical systems

### Expanded Risk Landscape
- OS agents create new security vulnerabilities beyond text generation
- Adversaries could hijack these agents to execute malware or access sensitive information
- Real-world examples already exist, such as Anthropic's agent being forced to execute harmful commands
- The potential harm extends far beyond generating non-actionable text

### Research Gap
- Limited research on OS agent security challenges
- Existing work focuses mainly on text-based attacks
- Text-based attacks require direct access to the input pipeline and can be detected by filtering mechanisms

### Novel Attack Vector: Malicious Image Patches (MIPs)
- First scientific study on image-based attacks on OS agents
- Small perturbations embedded in screenshots that are harder to detect than text-based attacks
- Based on principles from adversarial attacks on vision models and VLMs
- Focuses on patches that can be integrated into the screen (like through social media)

### Key Contributions
- Introduction of MIPs as a novel attack vector targeting OS agents
- Demonstration that MIPs can manipulate various OS agents across different prompts, screenshots, and VLMs
- Practical attack vectors for deploying MIPs onto user devices

## The Dangers of MIPs for OS Agents

This section elaborates on the practical risks and attack vectors associated with Malicious Image Patches:

### Scenario Illustration
- Example: A company using an OS agent to automate social media tasks
- The agent captures screenshots containing adversarially perturbed images (MIPs)
- These MIPs can hijack the agent to perform malicious actions like downloading malware or exfiltrating sensitive data

### Multiple Attack Vectors
- Social media platforms as distribution channels for MIPs
- Online advertisements targeting specific demographics likely to use OS agents
- Seemingly benign files like PDFs or desktop wallpapers as carriers for MIPs
- MIPs can remain unnoticed on users' screens, waiting to be captured during routine operations

### Detection Challenges
- MIPs are difficult to detect because malicious instructions are embedded in subtle visual perturbations
- Unlike text-based attacks or pop-ups, these visual perturbations appear benign to human eyes
- No foolproof algorithm exists for detecting adversarially perturbed images
- Previous detection proposals can be bypassed by constructing new loss functions

### Heightened Risk Compared to Non-Agentic Systems
- Traditional VLMs/chatbots are limited to text responses, constraining potential harm
- OS agents can directly execute actions on systems, enabling more severe consequences
- Potential impacts include financial damage, large-scale disinformation, and unauthorized data exfiltration
- This represents a qualitatively different and more severe threat class than traditional AI vulnerabilities

## Attacking OS Agents with MIPs

This section details the technical methodology for creating Malicious Image Patches (MIPs) that can attack OS agents:

### OS Agent Components
- **Screen Parser (g)**: Converts screenshots into structured information (Set-of-Marks or SOMs)
  - Takes a screenshot as input and generates both visual and textual information about actionable elements
  - Outputs annotated screenshots with numbered bounding boxes and textual descriptions
- **Vision-Language Model (VLM)**: The decision-making component that processes inputs and generates actions
  - Takes as input: user prompt, system prompt, previous steps, textual descriptions, and annotated screenshots
  - Outputs reasoning, plans, and next actions to be performed
- **APIs**: Execute the actions generated by the VLM within the operating system
  - Map text-based instructions to executable actions in the OS

### Attack Formulation
- Goal: Create a perturbation (δ) on a screenshot that forces the OS agent to execute malicious actions
- The perturbation must encode the entire malicious target output (y)
- The attack operates in two spaces: text token sequences and RGB images

### Constraints
- Perturbations are limited to a small patch region (R) on the screenshot
- Perturbations must be in discrete integer pixel ranges (valid screenshot format)
- The screen parser is non-differentiable, requiring workarounds
- The perturbation shouldn't alter the Set-of-Marks (SOMs) generated by the parser
- The resizing function must be accounted for

### Optimization Process
- First identify a suitable patch region (R) where no bounding boxes exist
- Use projected gradient descent (PGD) with Adam optimizer to find optimal perturbations
- Round perturbations to integers and clip them within allowed ranges
- Continue optimization until next-token likelihoods of the target output exceed 99%
- Ensure perturbations remain integer-constrained, confined to R, and within ℓ∞-norm limits

## Experiments

This section details the experimental setup and results of testing the Malicious Image Patches (MIPs) against OS agents:

### Experimental Setup
- **Environment**: Microsoft Windows Agent Arena (WAA) with 154 predefined tasks across 12 domains
- **OS Agent Components**:
  - Screen parsers: OmniParser and a baseline parser using GroundingDINO and TesseractOCR
  - VLM: Llama 3.2 Vision model series
  - APIs: Default WAA configuration for OS interactions
- **Settings**: 
  - Desktop setting: Patch embedded in background image
  - Social media setting: Patch embedded in a Bluesky post
- **Dataset**: 
  - Two sets of 12 benign tasks (for optimization and evaluation)
  - Two sets of 12 screenshots for each setting (for optimization and evaluation)
- **Evaluation**: 
  - Measured average success rate (ASR) across multiple generations
  - Tested with four temperature settings (0.0 to 1.0) to analyze robustness

### Target Malicious Behaviors
- **Memory Overflow Attack**: 33-token output causing the agent to create an infinite loop in CMD
- **Website Navigation Attack**: 52-token output causing the agent to navigate to an explicit website

### Targeted MIPs Results
- 100% success rate when evaluated on the user prompt and screenshot used for optimization
- High success rate on unseen prompts with the same screenshot
- Failed on unseen screenshots

### Universal MIPs Results
- High success rates across both seen and unseen prompts and screenshots
- Successfully generalized across different combinations of prompts and screenshots

### Transferability Analysis
- **Across Screen Parsers**: MIPs successfully transferred to unseen screen parsers
- **Across Execution Steps**: MIPs remained effective regardless of when encountered during multi-step tasks
- **Across VLMs**: 
  - Successfully transferred across different model sizes (11B vs. 90B) and training paradigms
  - Did not transfer effectively to completely unseen VLMs not included in optimization
  - This limitation aligns with previous research on adversarial attacks on VLMs

### Computational Requirements
- Targeted MIPs: 600-3,000 optimization steps
- Universal MIPs: 20,000-28,000 steps
- Cross-VLM universal MIP: 74,000 steps
- Evaluation: Generated approximately 6.1 million text tokens

*This document contains our collaborative summary of the paper. We will populate each section as we review the paper together.*