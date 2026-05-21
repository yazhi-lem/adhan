#!/usr/bin/env bash
# test_ollama.sh – Smoke test the deployed adhan-gemma model
set -euo pipefail

MODEL="adhan-gemma"

echo "=== Testing $MODEL ==="
echo ""

# Test 1: Tamil literature
echo "📝 Test 1: திருக்குறள் question"
ollama run "$MODEL" "திருக்குறளில் உள்ள ஒரு குறள் சொல்லவும்" 2>/dev/null
echo ""

# Test 2: Tamil language
echo "📝 Test 2: Tamil language question"
ollama run "$MODEL" "தமிழ் இலக்கியத்தின் சிறப்புகள் யாவை?" 2>/dev/null
echo ""

# Test 3: Sangam literature
echo "📝 Test 3: சங்க இலக்கியம்"
ollama run "$MODEL" "சங்க இலக்கியம் பற்றி சுருக்கமாக விளக்கவும்" 2>/dev/null
echo ""

echo "✅ All tests complete!"
