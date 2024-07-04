import numpy as np
labels_word_classes = ["wall", "barrier", "shop", "store", "boat", "ship", "father", "dad", "sheep", "lamb",
                  "ocean", "sea", "pants", "trousers", "gift", "present", "cash", "money", "road", "street",
                  "car", "automobile", "test", "exam", "war", "score", "coat", "dead", "lab", "key",
                  "browsers", "presence", "cat", "stream", "card", "taste", "hey", "google", "how", "is", "the", "weather",
                  "today", "what", "time", "it", "take", "a", "picture", "turn", "flashlight", "on", "I", "need", "new", "and",
                  "water", "resistant", "my", "Christmas", "so", "pretty", "at", "from", "lay", "beneath", "turned", "left",
                  "right", "with", "inside", "of", "behind", "was", "clear", "can", "help", "you", "get", "your", "tasted", "to",
                  "good", "taking", "find", "ingredients", "takes", "in",]


aligned_word_classes = ['barrier', 'wall', 'war','shop', 'store', 'score','ship',
                   'boat', 'coat','father', 'dad', 'dead','sheep', 'lamb',
                   'lab','ocean', 'sea', 'key','pants', 'trousers',
                   'browsers','gift', 'present', 'presence','money',
                   'cash', 'cat','road', 'street', 'stream','automobile',
                   'car', 'card','exam', 'test', 'taste', 'hey', 'google',
                   'how', 'is', 'the',
                       'weather', 'today', 'what', 'time', 'it',
                       'take', 'a', 'picture', 'turn', 'flashlight',
                       'on', 'I', 'need', 'new', 'and', 'water',
                       'resistant', 'my', 'Christmas', 'so', 'pretty',
                       'at', 'from', 'lay', 'beneath', 'turned', 'left',
                       'right', 'with', 'inside', 'of', 'behind', 'was',
                       'clear', 'can', 'help', 'you', 'get', 'your',
                       'tasted', 'to', 'good', 'taking', 'find',
                       'ingredients', 'takes', 'in', ]

### control group
control_group = ['wall', 'store', 'boat', 'dad', 'lamb', 'sea', 'trousers',
                 'present', 'cash', 'street',
                 'car', 'test']
## semantic group
experimental_group = ['barrier', 'shop', 'ship', 'father', 'sheep', 'ocean',
                      'pants', 'gift', 'money', 'road',
                      'automobile', 'exam']
## phonological group
treatment_group = ['war', 'score', 'coat', 'dead', 'lab', 'key', 'browsers',
                   'presence', 'cat', 'stream',
                   'card', 'taste']

# ## get indices of each group
aligned_word_classes = np.array(aligned_word_classes)
control_group_indices = [np.where(aligned_word_classes == word)[0][0] for word in control_group]
experimental_group_indices = [np.where(aligned_word_classes == word)[0][0] for word in experimental_group]
treatment_group_indices = [np.where(aligned_word_classes == word)[0][0] for word in treatment_group]


sentences= [
'Hey Google',
'How is the weather today',
'What time is it',
'Take a picture',
'Turn the flashlight on',
'I need a new lab coat and water resistant pants',
'I need a new lab coat and water resistant trousers',
'My new Christmas gift is so pretty',
'My new Christmas present is so pretty',
'The dead at sea from the war lay beneath the ship',
'The dead at ocean from the war lay beneath the boat',
'The car turned left and right on the road',
'The automobile turned left and right on the street',
'I left a Christmas card with cash inside today',
'I left a Christmas card with money inside today',
'The presence of the cat behind the barrier was clear',
'The presence of the cat behind the wall was clear',
'Browsers can help you get a pretty good score on your exam',
'Browsers can help you get a pretty good score on your test',
'The sheep tasted the clear water from the stream',
'The lamb tasted the clear water from the stream',
'The key to a good taste is taking the time to find the right ingredients at the store',
'The key to a good taste is taking the time to find the right ingredients at the shop',
'My father takes a good picture in good weather',
'My dad takes a good picture in good weather']

nlp_color = (118 / 255, 119 / 255, 230 / 255)
mel_color = (227 / 255, 99 / 255, 207 / 255)
semantic_color = (20 / 255, 53 / 255, 111 / 255)
phonological_color = (234 / 255, 54 / 255, 64 / 255)