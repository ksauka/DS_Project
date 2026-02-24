---
title: Customer Service Assistant
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Customer Service Assistant

**Interactive intent classification system using hierarchical Dempster-Shafer theory for uncertain query resolution.**

## Overview

This application demonstrates a novel approach to intent classification in customer service:

- **Hierarchical Reasoning**: Uses Dempster-Shafer theory to handle uncertainty across parent-child intent hierarchies
- **Explainable AI**: Provides transparent belief updates and clarification strategies
- **Interactive Dialogue**: Engages users in multi-turn conversations to resolve ambiguous queries
- **Multi-Dataset Support**: Banking77, CLINC150, SNIPS, ATIS, TOPv2 with hierarchical intent structures

## Features

### 🎯 **Intent Classification**
- Support for multiple datasets: Banking77 (77 intents), CLINC150 (150 intents), SNIPS, ATIS, TOPv2
- Hierarchical classification with parent-child relationships
- Confidence thresholds for each hierarchy level

### 🤔 **Uncertainty Management**
- Dempster-Shafer belief propagation
- Ancestor-aware confidence scoring
- Intelligent clarification questions when belief is uncertain

### 📊 **Explainability**
- Real-time belief progression visualization
- Top-5 intent probabilities display
- Transparent decision-making process

### 📈 **Research Integration**
- Session logging to private GitHub repository
- Qualtrics/Prolific integration for user studies
- Dropbox model storage for large ML models
- Comprehensive interaction tracking

## Technical Architecture

### Core Components
- **Classifier**: Logistic Regression with sentence-transformers embeddings (intfloat/e5-base)
- **DS Reasoning**: Hierarchical belief propagation with configurable thresholds
- **Clarification**: Context-aware question generation using GPT-4
- **Data Logging**: GitHub API integration for session persistence

### Datasets
- **Banking77**: 77 banking intents across 13,083 queries
- **CLINC150**: 150 general customer service intents
- **SNIPS**, **ATIS**, **TOPv2**: Additional domain-specific datasets
- **Hierarchy**: Configurable 3-level structures (root → parent → leaf)
- **Split**: 80% train, 10% dev, 10% test

## Usage

### For Participants
1. Enter your Prolific ID (if coming from a study)
2. Interact with banking queries
3. Answer clarification questions when needed
4. View intent predictions with confidence scores
5. Complete post-study survey

### For Researchers
Configure secrets in Space Settings:
- `OPENAI_API_KEY`: For clarification generation
- `GITHUB_TOKEN`: For session logging
- `DROPBOX_*`: For model storage

## Research Context

Part of the HicXAI research project investigating explainable AI methods in hierarchical intent classification for customer service automation.

**Note**: This is a research prototype. Not intended for production customer service systems.

## License

MIT License - See LICENSE file for details

---

**Deployment**: This Space uses Docker SDK with custom configuration for model downloads and session logging.
