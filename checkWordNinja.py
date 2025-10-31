from wordninja import split as wordninja_split

# Check if wordninja works
test_words = ['bedroom', 'sunlight', 'football', 'daybed']
print("Wordninja splits:")
for word in test_words:
    print(f"  {word} → {wordninja_split(word)}")
