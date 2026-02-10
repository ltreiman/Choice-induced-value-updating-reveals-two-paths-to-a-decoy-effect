function calc_amount_pos(p) {
  return {left: 25*Math.cos((180*p - 90)*Math.PI/180), top: 25*Math.sin((180*p - 90)*Math.PI/180)}
}

function create_trials(trials, isPractice){
  // riskySafeOrders = ["SR","RS"];  
  riskySafeDecoyOrders = ["SRD", "SDR", "RSD", "RDS", "DSR", "DRS"];
  riskyDecoyOrder = shuffleArray(Array(trials.total/(2*6)).fill(riskySafeDecoyOrders).flat(),trials.total/12)
  safeDecoyOrder = shuffleArray(Array(trials.total/(2*6)).fill(riskySafeDecoyOrders).flat(),trials.total/12)
  // noDecoyOrder =  shuffleArray(Array(trials.total/(3*2)).fill(riskySafeOrders).flat(),trials.total/6)

  if (isPractice==1){
    array = Array.from({length: trials.practice}, (_, i) => i + 1).map(v => v+2);
  }
  else {
    array = shuffleArray(Array.from({length: trials.total}, (_, i) => i + 1), trials.total);
    randomTrials = random_trials(trials.total, trials.outcome)
  }
    stimuli = []
    for (let i=0; i < array.length; i++){
        trial_number = i+1; 
        trial_type = ""

        // Risky and safe gambles calculated the same way
        riskyGamble = {
          id: "R",
          p: Math.round(getRandomArbitrary(probabilities.risky[0],probabilities.risky[1])*100)/100,
          amount: Math.round(getRandomArbitrary(riskyAmount[0],riskyAmount[1])),
          opacity: non_selection_transparancy,
        }

        safeGamble = {
          id: "S",
          p: Math.round(getRandomArbitrary(probabilities.safe[0],probabilities.safe[1])*100)/100,
          opacity: non_selection_transparancy,
        }
        safeGamble["amount"] = Math.round(riskyGamble.p * riskyGamble.amount / safeGamble.p)
        if (safeGamble.amount<=0){ 
          safeGamble.amount=1
        }
        // For risky Decoy trials
        if(array[i]%2==1){
          trial_type = "riskyDecoy"
          order_gambles = riskyDecoyOrder.pop()
          decoyGamble= {
            p: riskyGamble.p - (Math.floor(getRandomArbitrary(decoyDifference.risky[0], decoyDifference.risky[1]))/100),
            id: "D",
            amount: riskyGamble.amount,
          }
        }
        // For safe decoy trials
        else {
          trial_type = "safeDecoy"
          order_gambles = safeDecoyOrder.pop()
          decoyGamble = {
            p: safeGamble.p,
            id: "D",
            amount: safeGamble.amount-Math.floor(getRandomArbitrary(decoyDifference.safe[0], decoyDifference.safe[1])),
          }
        }
        decoyGamble["opacity"] = non_selection_transparancy

        // Ensure that the gambles align with order_gambles
        let gambles = { "R": riskyGamble, "S": safeGamble, "D":  decoyGamble};
        shuffledGambles = order_gambles.split("").map(letter => gambles[letter]).filter(gamble => gamble !== undefined);
        if(shuffledGambles.length == 2){
          console.log(shuffledGambles)
        }
        
        stimuli.push({
            trialNumber: i+1,
            trial_set: trial_type,
            gambles: shuffledGambles,
            riskyGamble,
            decoyGamble,
            safeGamble,
            order_gambles,
            practice: isPractice,
            random_trial: isPractice == 1? false : randomTrials.includes(trial_number)
        })
    }
    return(stimuli)
}

function random_trials(total_trials, noutcomes){
  start = 1
  end = total_trials
  temp = [...Array(end - start + 1).keys()].map(x => x + start);
  randomTrials = jsPsych.randomization.sampleWithoutReplacement(temp, noutcomes);
  return randomTrials
}


  
function shuffleArray(array, l) {
    for (var i = l - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array
}
function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
  }

  function calculate_winnings(trial){
    if(trial.final_response == null){
      return 0
    }
    p_random = Math.random() // Draw random p
    return(p_random <= trial.final_response_probability ? trial.final_response_amount : 0)
  }