from rouge import Rouge
"""
hypothesis0 = "tired after a long journey from scotland , nikki kelly decided to have a quiet night in ."
hypothesis1 = "she started to feel cramps in her stomach , and believed she might be getting her period ."

hypothesis2 = "but when she made a dash for the bathroom , the 24-year-old was shocked to find she was in fact going into the early stages of labour , and gave birth to her son on the bathroom floor ."
hypothesis3 = "miss kelly , from bridport , dorset , had no idea she was expecting , and had even kept her size eight figure throughout the pregnancy ."

hypothesis4 = "nikki kelly believed she was suffering from period cramps - before giving birth to son james , now three months , in ` three pushes ' on the bathroom floor ."

hypothesis5 = "miss kelly kept her size 8 figure throughout her pregnancy , and had no cravings . she is pictured on holiday with partner andrew swallow , 27 , when she was unknowingly four months pregnant ."

hypothesis6 = "she had been taking the contraceptive pill and continued to have periods for the whole nine months -- and experienced no morning sickness or cravings whatsoever ."
hypothesis7 = "she says baby james was born in ` three pushes ' and she and her partner , aaron swallow , 27 , have now made an offer on a house so they can all live together as a family ."
hypothesis8 = "miss kelly , who worked as a cleaner before her pregnancy , said : ` i was feeling a bit rubbish with what i thought was cramps so decided to have a quiet night in after travelling back from scotland and seeing my family ."
hypothesis9 = "` i started needing the toilet more and more often , until i could n't even get off the bathroom floor ."

reference1 =  "nikki kelly , 24 , kept needing the toilet and believed she had period cramps ." \
              "she was actually in labour and gave birth to her son on the bathroom floor ." \
              "her pregnancy came as a shock as she had been on the contraceptive pill ." \
              "she continued to have periods , and had no baby bump and no cravings ."

hypothesis10 = "manchester united star cristiano ronaldo has completed a notable awards double after being named player of the year by the english football writers ' association for the second year in a row ."


ref = "new : cristiano ronaldo wins english writers ' award for second year in a row ."

"""

#hypothesis9721_0 = "archie brown struck and killed a 2-year-old who ran in front of his van in milwaukee , wisconsin sunday evening .the car accident occurred at 5:10 pm sunday after the child ran into the street .the driver stayed in his van while police arrived .a 15-year-old boy was also shot and later died .milwaukee police say the driver of a van who struck a 2-year-old has been shot dead at the scene of the accident ."
#hypothesis9721_1 = "the car accident occurred at 5:10 pm sunday .archie brown jr killed a 2-year-old who ran in front of his van in milwaukee , wisconsin sunday evening .police say no suspects have been brought into custody .milwaukee police say the driver of a van who struck and killed a 2-year-old has been shot dead at the scene of the accident .a 15-year-old boy was also shot and later died .police say the teen was not a passenger in the van ."
#hypothesis9721_2 = "archie brown jr killed a 2-year-old who ran in front of his van in milwaukee , wisconsin sunday evening .the car accident occurred at 5:10 pm sunday .police say no suspects have been brought into custody .a 15-year-old boy was also shot and later died .the driver of a van who struck and killed a 2-year-old has been shot dead at the scene of the accident ."
#ref = "a 41-year-old driver hit and killed a 2-year-old girl sunday evening when the toddler ran front of his van in a milwaukee neighborhood .the driver , archie brown jr. , stopped at the scene .moments later , someone from the home where the toddler lived opened fire - shooting brown dead and fatally wounding a 15-year-old .police are investigating whether the shooting was revenge for the fatal traffic accident .the gunman responsible has not been arrested .brown was a father of four with a young daughter of his own ."

rouge = Rouge()
#scores9721_0 = rouge.get_scores(hypothesis9721_0, ref)
#scores9721_1 = rouge.get_scores(hypothesis9721_1, ref)
#scores9721_2 = rouge.get_scores(hypothesis9721_2, ref)
#print(scores9721_0)
#print(scores9721_1)
#print(scores9721_2)


#hypothesis139_0 = "passenger says a stranger sitting behind him tried to choke him .oliver minatel , 22 , was sleeping on air canada flight 8623 from toronto .minatel was traveling with his teammates from the ottawa fury football club .`` with a rope , something that he has , he just jumped on me , '' minatel says ."
#hypothesis139_1 = "oliver minatel , 22 , said he was sleeping on air canada flight 8623 from toronto .a stranger sitting behind him tried to choke him .the incident occurred about a half-hour before the flight landed .minatel : `` with a rope , something that he has , he just jumped on me ''"
#ref = "oliver minatel , a 22-year-old player from brazil , was attacked from behind , he says .witnesses say suspect tried to choke him with the cord from his headphones .team says forward is ok , will play saturday night ; suspect was taken for evaluation ."
#rouge = Rouge()
#scores139_0 = rouge.get_scores(hypothesis139_0, ref)
#scores139_1 = rouge.get_scores(hypothesis139_1, ref)
#scores139_2 = rouge.get_scores(hypothesis139_2, ref)
#print(scores139_0)
#print(scores139_1)

file_name = "9824"

f_ref = open('../../../../data/smy/decompressed/data/refs/test/'+file_name+".ref")

final_ref = f_ref.read()


f_ours = open('../../../../data/smy/fast/save_a2_a5/output/'+file_name+".dec")

final_ours = f_ours.read()

f_baseline = open('../../../../data/smy/fast/save_baseline/decoded/output/'+file_name+".dec")

final_baseline = f_baseline.read()

print("our")
scores_ours = rouge.get_scores(final_ours, final_ref)

scores_baseline = rouge.get_scores(final_baseline, final_ref)


print(scores_ours)
print("baseline")
print(scores_baseline)