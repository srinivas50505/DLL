text="welcome to dl lab"
words=text.split()
vocabulary=sorted(set(words))
onehotencoding=[]
for word in vocabulary:
    encoding =[0]*len(vocabulary)
    index=list(vocabulary).index(word)
    encoding[index]=1
    onehotencoding.append((word,encoding))
print("Original text: ",text)
for word,encoding in onehotencoding:
    print(f"{word} : {encoding}" )
