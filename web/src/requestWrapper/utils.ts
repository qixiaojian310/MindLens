// 定义症状常量
export const SYMPTOMS = {
  Thedisturbanceisnotsubstanceinduced: 'The disturbance is not due to substance use, medications, or other medical conditions.',
  Increaseingoaldirectedactivity: 'A noticeable increase in energy, focus, or productivity towards achieving specific goals.',
  Restlessness: 'An inability to remain still or calm, often accompanied by fidgeting or pacing.',
  X6monthduration: 'The symptoms or condition must persist for at least six months to meet diagnostic criteria.',
  Dissociativereaction: 'A psychological response characterized by feeling detached from reality or oneself.',
  Psychomotoragitation: 'Excessive movement or fidgeting that is purposeless and often driven by inner tension.',
  Hypervigilance: 'A state of increased alertness and sensitivity to potential dangers or threats.',
  Thedisturbancecausesclinicallysignificantdistress: 'The symptoms cause substantial impairment in social, occupational, or other important areas of functioning.',
  Lossofinterestorpleasureinactivities: 'Reduced interest or pleasure in most or all daily activities that were once enjoyable.',
  Lackofsleeporoversleeping: 'Difficulty falling asleep, staying asleep, or oversleeping, often disrupting daily functioning.',
  Intrusivememoriesorflashbacks: 'Recurrent, unwanted memories or vivid reliving of a traumatic event.',
  Experiencingtraumaticevent: 'Direct exposure to a distressing or life-threatening situation.',
  Persistentsadnessorlowmood: 'A prolonged period of low mood or emotional numbness.',
  Witnessingtraumaticevent: 'Observing a traumatic event as it happens to others.',
  Hallucinations: 'Experiencing sensory perceptions, such as hearing voices or seeing things, that are not present.',
  Exaggeratedstartleresponse: 'Overreacting to unexpected stimuli, often with heightened fear or surprise.',
  Depressedmood: 'A state of feeling consistently low or hopeless.',
  Irritability: 'Increased sensitivity to frustration or anger, often leading to outbursts.',
  Moretalkativethanusual: 'Speaking significantly more than usual or feeling compelled to keep talking.',
  Angryoutburst: 'Sudden and intense expressions of anger, often disproportionate to the situation.',
  X1monthduration: 'The symptoms or condition must persist for at least one month to meet diagnostic criteria.',
  Feelingofdetachment: 'A sense of being emotionally or physically disconnected from others.',
  Diminishedinterest: 'A noticeable decrease in enthusiasm or engagement with previously enjoyed activities.',
  Fatigueorlossofenergy: 'Persistent tiredness or lack of energy, even after rest.',
  Morethanonemonthofdisturbance: 'Symptoms or disturbance lasting more than one month.',
  Racingthoughts: 'A rapid flow of thoughts that can feel overwhelming and disorganized.',
  Persistentnegativeemotionalstate: 'Ongoing negative feelings such as fear, guilt, or anger.',
  Excessiveinvolvementinactivitieswithhighpotentialforpainfulconsequences: 'Engaging in behaviors with a high likelihood of adverse outcomes, driven by impulsivity.',
  Diminishedemotionalexpression: 'Reduced ability to express emotions or facial responses.',
  Catatonicbehavior: 'A state of immobility or lack of responsiveness, often associated with certain psychiatric conditions.',
  Recurrentdistressingdreamingaffiliatedwiththetraumaticevent: 'Frequent nightmares or disturbing dreams related to a traumatic experience.',
  Recklessness: 'Careless behavior that disregards personal or others\' safety.',
  Intensedistressorreactionwhenexposedtocuesaffiliatedwiththetraumaticevent: 'Severe emotional or physical reaction to reminders of a traumatic event.',
  Persistentinabilitytoexperiencepositiveemotions: 'Difficulty experiencing happiness or satisfaction, even in positive situations.',
  Sleepdisturbance: 'Difficulty falling asleep, staying asleep, or oversleeping, often disrupting daily functioning.',
  Persistentandexaggeratednegativebeliefaboutoneselfortheworld: 'Strong, lasting negative thoughts about oneself, others, or the world.',
  Delusions: 'Fixed, false beliefs that are resistant to reason or contrary evidence.',
  Inflatedselfesteem: 'Exaggerated sense of one\'s abilities, importance, or achievements.',
  Disorganizedthinkingorspeech: 'Confused or illogical thought patterns that affect communication.',
  Excessiveworryorfear: 'Constant or exaggerated fear about potential negative outcomes.',
  Persistentlossofmemoryaboutthecauseorconsequencesofthetraumaticevent: 'Inability to recall significant details about a traumatic event or its aftermath.',
  Difficultyconcentratingormakingdecisions: 'Trouble focusing or making decisions, often affecting daily activities.',
  Weightlossorgain: 'Noticeable and unintended weight loss or gain.',
  Thoughtsofsuicide: 'Recurring thoughts of death or self-harm.',
  Avoidanceoftraumaticevent: 'Deliberate efforts to avoid reminders, thoughts, or feelings related to a traumatic experience.',
} as const

// 定义类型
export type SymptomKey = keyof typeof SYMPTOMS

// 定义函数
export function convertSymptom(key: SymptomKey): string {
  return SYMPTOMS[key]
}
