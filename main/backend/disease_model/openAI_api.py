import os
import base64
from openai import AzureOpenAI
from disease_model.configopenAI import AzureOpenAIConfig  # Make sure the file is named 'openAIconfig.py'

#AzurOpenAIConfig definition create a file that have this class named configopenAI.py and change the parametre
# class AzureOpenAIConfig:  
#     def __init__(self):  
#         self.endpoint = os.getenv("ENDPOINT_URL", "https://openai-api-pcd-sym.openai.azure.com/")  
#         self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")  
#         self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")  
      
#     def get_endpoint(self):  
#         return self.endpoint  
  
#     def get_deployment(self):  
#         return self.deployment  
  
#     def get_subscription_key(self):  
#         return self.subscription_key  


class AzureAssistant:
    def __init__(self, config: AzureOpenAIConfig):
        self.client = AzureOpenAI(
            azure_endpoint=config.get_endpoint(),
            api_key=config.get_subscription_key(),
            api_version="2025-01-01-preview"
        )
        self.model = config.get_deployment()


    def ask(self, user_message):
        if user_message.lower().startswith("extract symptoms"):
            symptom_extraction_prompt = """You are an intelligent medical NLP assistant. Your task is to extract only the symptoms from the following patient description.Use **only** the predefined list of symptoms below. Return a clean list of the symptoms **exactly as they appear**, separated by '#', without explanations or extra text.
                                        Here is the list of valid symptoms: [INSERT_SYMPTOM_LIST_HERE]
                                        Now extract the symptoms present in the following text and return only a comma-separated list of matched symptoms from the above list."""
            
            patient_description = user_message[len("extract symptoms"):].strip()
            system_prompt = symptom_extraction_prompt.replace("[INSERT_SYMPTOM_LIST_HERE]", "anxiety and nervousness,depression,shortness of breath,depressive or psychotic symptoms,sharp chest pain,dizziness,insomnia,abnormal involuntary movements,chest tightness,palpitations,irregular heartbeat,breathing fast,hoarse voice,sore throat,difficulty speaking,cough,nasal congestion,throat swelling,diminished hearing,lump in throat,throat feels tight,difficulty in swallowing,skin swelling,retention of urine,groin mass,leg pain,hip pain,suprapubic pain,blood in stool,lack of growth,emotional symptoms,elbow weakness,back weakness,symptoms of the scrotum and testes,swelling of scrotum,pain in testicles,flatulence,pus draining from ear,jaundice,mass in scrotum,white discharge from eye,irritable infant,abusing alcohol,fainting,hostile behavior,drug abuse,sharp abdominal pain,feeling ill,vomiting,headache,nausea,diarrhea,vaginal itching,vaginal dryness,painful urination,involuntary urination,pain during intercourse,frequent urination,lower abdominal pain,vaginal discharge,blood in urine,hot flashes,intermenstrual bleeding,hand or finger pain,wrist pain,hand or finger swelling,arm pain,wrist swelling,arm stiffness or tightness,arm swelling,hand or finger stiffness or tightness,wrist stiffness or tightness,lip swelling,toothache,abnormal appearing skin,skin lesion,acne or pimples,dry lips,facial pain,mouth ulcer,skin growth,eye deviation,diminished vision,double vision,cross-eyed,symptoms of eye,pain in eye,eye moves abnormally,abnormal movement of eyelid,foreign body sensation in eye,irregular appearing scalp,swollen lymph nodes,back pain,neck pain,low back pain,pain of the anus,pain during pregnancy,pelvic pain,impotence,infant spitting up,vomiting blood,regurgitation,burning abdominal pain,restlessness,symptoms of infants,wheezing,peripheral edema,neck mass,ear pain,jaw swelling,mouth dryness,neck swelling,knee pain,foot or toe pain,bowlegged or knock-kneed,ankle pain,bones are painful,knee weakness,elbow pain,knee swelling,skin moles,knee lump or mass,weight gain,problems with movement,knee stiffness or tightness,leg swelling,foot or toe swelling,heartburn,smoking problems,muscle pain,infant feeding problem,recent weight loss,problems with shape or size of breast,difficulty eating,scanty menstrual flow,vaginal pain,vaginal redness,vulvar irritation,weakness,decreased heart rate,increased heart rate,bleeding or discharge from nipple,ringing in ear,plugged feeling in ear,itchy ear(s),frontal headache,fluid in ear,neck stiffness or tightness,spots or clouds in vision,eye redness,lacrimation,itchiness of eye,blindness,eye burns or stings,itchy eyelid,feeling cold,decreased appetite,excessive appetite,excessive anger,loss of sensation,focal weakness,slurring words,symptoms of the face,disturbance of memory,paresthesia,side pain,fever,shoulder pain,shoulder stiffness or tightness,shoulder weakness,shoulder swelling,tongue lesions,leg cramps or spasms,ache all over,lower body pain,problems during pregnancy,spotting or bleeding during pregnancy,cramps and spasms,upper abdominal pain,stomach bloating,changes in stool appearance,unusual color or odor to urine,kidney mass,swollen abdomen,symptoms of prostate,leg stiffness or tightness,difficulty breathing,rib pain,joint pain,muscle stiffness or tightness,hand or finger lump or mass,chills,groin pain,fatigue,abdominal distention,regurgitation.1,symptoms of the kidneys,melena,flushing,coughing up sputum,seizures,delusions or hallucinations,pain or soreness of breast,excessive urination at night,bleeding from eye,rectal bleeding,constipation,temper problems,coryza,wrist weakness,hemoptysis,lymphedema,skin on leg or foot looks infected,allergic reaction,congestion in chest,muscle swelling,low back weakness,sleepiness,apnea,abnormal breathing sounds,excessive growth,blood clots during menstrual periods,absence of menstruation,pulling at ears,gum pain,redness in ear,fluid retention,flu-like syndrome,sinus congestion,painful sinuses,fears and phobias,recent pregnancy,uterine contractions,burning chest pain,back cramps or spasms,stiffness all over,'muscle cramps, contractures, or spasms',low back cramps or spasms,back mass or lump,nosebleed,long menstrual periods,heavy menstrual flow,unpredictable menstruation,painful menstruation,infertility,frequent menstruation,sweating,mass on eyelid,swollen eye,eyelid swelling,eyelid lesion or rash,unwanted hair,symptoms of bladder,irregular appearing nails,itching of skin,hurts to breath,'skin dryness, peeling, scaliness, or roughness',skin on arm or hand looks infected,skin irritation,itchy scalp,incontinence of stool,warts,bumps on penis,too little hair,foot or toe lump or mass,skin rash,mass or swelling around the anus,ankle swelling,drainage in throat,dry or flaky scalp,premenstrual tension or irritability,feeling hot,foot or toe stiffness or tightness,pelvic pressure,elbow swelling,early or late onset of menopause,bleeding from ear,hand or finger weakness,low self-esteem,itching of the anus,swollen or red tonsils,irregular belly button,lip sore,vulvar sore,hip stiffness or tightness,mouth pain,arm weakness,leg lump or mass,penis pain,loss of sex drive,obsessions and compulsions,antisocial behavior,neck cramps or spasms,poor circulation,thirst,sneezing,bladder mass,premature ejaculation,leg weakness,penis redness,penile discharge,shoulder lump or mass,cloudy eye,hysterical behavior,arm lump or mass,nightmares,bleeding gums,pain in gums,bedwetting,diaper rash,lump or mass of breast,vaginal bleeding after menopause,itching of scrotum,postpartum problems of the breast,hesitancy,muscle weakness,throat redness,joint swelling,redness in or around nose,wrinkles on skin,foot or toe weakness,hand or finger cramps or spasms,back stiffness or tightness,wrist lump or mass,skin pain,low urine output,sore in nose,ankle weakness")
            user_content = patient_description
        elif user_message.startswith("+"):
            system_prompt = "You are a helpful AI assistant.that help with any thing, Add clear line breaks (`\\n`) for formatting your reasoning to improve readability."
            user_content = user_message[1:].strip()
        else:
            system_prompt = "You are a medical assistant that takes the description of the patient and the AI-predicted disease, and generates a helpful and clear response for the patient. Make the prompt short and don't give suggestions."
            user_content = user_message

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_content
                    }
                ]
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"



# ai = AzureAssistant()
# response = ai.ask("i have depression and groim mas the prediction that may model did is panic dessorder")
# print("AI Response:", response)

