"""Evaluation module for tournament analysis."""

from evaluations.base_evaluator import BaseEvaluator, GameResult
from evaluations.evaluate_werewolf import WerewolfEvaluator
from evaluations.evaluate_avalon import AvalonEvaluator
from evaluations.evaluate_sheriff import SheriffEvaluator
from evaluations.evaluate_spyfall import SpyfallEvaluator
from evaluations.evaluate_amongus import AmongUsEvaluator

__all__ = [
    'BaseEvaluator',
    'GameResult',
    'WerewolfEvaluator',
    'AvalonEvaluator',
    'SheriffEvaluator',
    'SpyfallEvaluator',
    'AmongUsEvaluator',
]

