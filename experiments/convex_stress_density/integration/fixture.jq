# Synthetic scores and identity sets only; no original examples or discovery.
def candidate($id;$ce;$accuracy): {id:$id,train_ce:$ce,vocab:97,model_sha256:"fixture-model",
  primary_pass:($accuracy>=0.99),all_splits_pass:($accuracy>=0.99),
  audit:{train_ce:$ce,vocabulary:97},metrics:{train:{loss:$ce,sequence_accuracy:0.5},validation:{loss:$ce,sequence_accuracy:$accuracy}}};
$protocol[0] as $p
| ($suite[0]|.tasks|=map(.selection="validation"|.data_fingerprints={train:"fixture-train",validation:"fixture-validation"})) as $suite
| [$suite.tasks[].name as $task|$p.modes[]|[$task,.]] as $all
| [$p.tasks[] as $task|$p.modes[]|[$task,.]] as $pairs
| {tasks:[$suite.tasks[]|{task:.name,vocab:97,scale:1,data_fingerprints:.data_fingerprints}]} as $inventory
| {choices:[$all[]|{task:.[0],arm:.[1],status:"selected",tier:"fixture",selection:"validation",training_ce:4.2,
    primary_pass:false,all_splits_pass:false,metrics:candidate("hull";4.2;0.5).metrics}]} as $hull
| {parents:{full_suite_hull:$hull,hull_inventory:$inventory},choices:[$pairs[]|
    {task:.[0],training_mode:.[1],status:"selected",selected:candidate("general";3.9;0.5)}]} as $general
| [$p.families|to_entries[]|.value as $pin|.key as $index
    | ([4.0,4.1,3.8,3.7][$index]) as $ce
    | {name:$pin.name,summary:{source_fingerprint:$pin.source_fingerprint,producer_commit:$pin.producer_commit,
        tasks:9,process_failures:0,records:[$p.tasks[] as $task|
          {task:$task,result:{metadata:{source_fingerprint:$pin.source_fingerprint,producer_commit:$pin.producer_commit,
            training_gradient_evaluations:0,input:{task:$task,vocab:97,selection:"validation",data_fingerprints:{train:"fixture-train",validation:"fixture-validation"}}},
            construction_seconds:1,evaluation:{status:"complete",selected:"fresh",candidates:[candidate("fresh";$ce;0.5)]}}}],
        mode_references:[$pairs[]|{task:.[0],mode:.[1],standalone_constructor_reference:true,gradient_training_run:false,
            selected:candidate("fresh";$ce;0.5)}]}}] as $families
| {protocol:$p,suite:$suite,general:$general,families:$families}
