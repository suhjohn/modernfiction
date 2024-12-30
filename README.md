# Modern Fiction

With a single command, this tool will transform your old-fashioned fiction into modern fiction. 


From:
```
Sure he that made us with such large discourse,
Looking before and after, gave us not
That capability and godlike reason
To fust in us unus’d. Now whether it be
Bestial oblivion, or some craven scruple
Of thinking too precisely on th’event,—
A thought which, quarter’d, hath but one part wisdom
And ever three parts coward,—I do not know
Why yet I live to say this thing’s to do,
Sith I have cause, and will, and strength, and means
To do’t. Examples gross as earth exhort me,
Witness this army of such mass and charge,
Led by a delicate and tender prince,
Whose spirit, with divine ambition puff’d,
Makes mouths at the invisible event,
Exposing what is mortal and unsure
To all that fortune, death, and danger dare,
Even for an eggshell. Rightly to be great
Is not to stir without great argument,
But greatly to find quarrel in a straw
When honour’s at the stake. How stand I then,
That have a father kill’d, a mother stain’d,
Excitements of my reason and my blood,
And let all sleep, while to my shame I see
The imminent death of twenty thousand men
That, for a fantasy and trick of fame,
Go to their graves like beds, fight for a plot
Whereon the numbers cannot try the cause,
Which is not tomb enough and continent
To hide the slain? O, from this time forth,
My thoughts be bloody or be nothing worth.
```

To: 
```
Surely the one who made us with such broad understanding,
able to consider the past and the future, didn't give us
that capacity and divine reason
to rot inside us unused. Now, whether it's
mindless forgetfulness, or some cowardly hesitation
from thinking too precisely about the outcome—
a thought process that, when analyzed, has only one part wisdom
and three parts cowardice—I don't know
why I'm still alive to say this needs to be done,
since I have reason, will, strength, and means
to do it. Obvious examples urge me on,
like this army of such size and expense,
led by a refined and sensitive prince,
whose spirit, fueled by noble ambition,
mocks the unseen outcome,
risking everything mortal and uncertain
to all that fortune, death, and danger can do,
even for something trivial. True greatness
isn't about acting without good reason,
but about finding great cause for a fight in even the smallest thing
when honor is at stake. So where do I stand,
with a father murdered, a mother disgraced,
provocations to my reason and my emotions,
and yet I do nothing, while to my shame I watch
the coming deaths of twenty thousand men
who, for an illusion and a fleeting moment of fame,
go to their graves as if they were beds, fighting for a piece of land
so small it can't contain their numbers,
which isn't big enough to hold and bury the slain? Oh, from this moment on,
let my thoughts be bloody or be worthless.
```


## Requirements

- Python 3.12
- Poetry
- Together API key

## How to use
You need at least Together's API key.


0. Prepare an epub file that you want to transform. 
1. Create a file named `.env` in the root directory of the project with the following content:
```
GEMINI_API_KEY=<your gemini api key>
TOGETHER_API_KEY=<your together api key>
OPENAI_API_KEY=<your openai api key>
ANTHROPIC_API_KEY=<your anthropic api key>
DEEPSEEK_API_KEY=<your deepseek api key>
```

2. Install dependencies of the project. 
```
poetry install
```
3. Bring in an epub file to the directory.
4. Run the script with the right arguments. 
```
poetry run python modernfiction/main.py -i <path to your epub file> -o <path to your output epub file>
```

For example, if you downloaded a file named hamlet.epub and you moved it to this directory, you can run the script like this: 
```
PYTHONPATH=. poetry run time python  modernfiction/main.py -i ./hamlet.epub -o ./hamlet_new.epub -p gemini -m gemini-1.5-flash-8b
```
