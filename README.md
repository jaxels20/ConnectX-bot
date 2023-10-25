# Connect4 Agents with Advanced Algorithms

Welcome to my repository showcasing three distinct agents designed to play the game of Connect4. These agents demonstrate a range of Machine Learning and AI algorithms, from classic Minimax to an AlphaZero-inspired MCTS with a Neural Network.

## Introduction

Connect4 is a two-player connection board game, where players choose a color and then take turns dropping one colored disc from the top into a vertically suspended grid. The pieces fall straight down, occupying the lowest available space within the column. The object of the game is to connect four of one's own discs of the same color next to each other vertically, horizontally, or diagonally before your opponent.

## Agents Overview

### Minimax Agent

The Minimax agent is based on the classic algorithm where it explores all possible moves to a certain depth and evaluates the game state. It uses a heuristic evaluation function to gauge the strength of a particular board state and makes decisions accordingly.

### MCTS with Neural Network (AlphaZero-like)

Inspired by the techniques used in AlphaZero, this agent employs the Monte Carlo Tree Search (MCTS) guided by a Neural Network. The neural network is trained to predict the likely outcomes of different moves and guide the search process more efficiently, enabling it to make informed decisions about the best moves to play.

### Pure MCTS Agent

This agent utilizes a pure Monte Carlo Tree Search without the guidance of a neural network. It relies on extensive sampling and exploration of possible moves to determine the best action to take, offering a blend of randomness and strategy in its gameplay.

.
