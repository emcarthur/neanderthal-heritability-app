if (!window.dash_clientside) {
    window.dash_clientside = {}
}
window.dash_clientside.clientside = {
    update_figure: function(indexes, store) {
      var sim_idxs = indexes[0]
      var colors = indexes[1]
      var fig = store;
      var new_fig = {};

        new_fig['layout'] = fig['layout'];
        new_fig['data'] = [];
        let fitness_idx = [0, 1, 5, 9, 20, 25]
        sim_idxs.forEach(function(item, index) { //for (i = 0; i < 3; i++) { // fig['data'].length
		        new_fig['data'].push(fig['data'][item]);
            new_fig['data'][index]['line']['color']= colors[index]
        });
      return new_fig;
    }
};

window.dash_clientside.clientside2 = {
    update_figure2: function(inputValues,store) {
      var y_dist_vals = inputValues[0];
      var traitAssociation = inputValues[1];
      var colors = inputValues[2];
      var new_fig = {};
      var fig = store;
      new_fig['layout'] = fig['layout'];
      new_fig['data'] = fig['data'].slice(0,3);

      new_fig['data'][0]['y'] = y_dist_vals;
      new_fig['data'][1]['y'] = y_dist_vals.slice(0,31);
      new_fig['data'][2]['y'] = y_dist_vals.slice(69,100);
      //for (i=0;i<traitAssociation.length,i++){

      new_fig['data'][0]['y'] = y_dist_vals;
      new_fig['data'][1]['y'] = y_dist_vals.slice(0,31);
      new_fig['data'][2]['y'] = y_dist_vals.slice(69,100);
      traitAssociation.forEach(function(traitAssociation_y, i) {
        var new_box = {
              "boxpoints": "all",
              "fillcolor": "rgba(255,255,255,0)",
              "hoverinfo": "skip",
              "jitter": 0,
              "line": {
                "color": "rgba(255,255,255,0)"
              },
              "marker": {
                "color": colors[i],
                "symbol": "line-ns-open"
              },
              "showlegend": false,
              "x": [
                traitAssociation_y
              ],
              "y": [
                0
              ],
              "type": "box",
              "xaxis": "x2",
              "yaxis": "y2"
            };
        new_fig['data'].push(new_box);
      });

      new_fig['layout']['yaxis2']['ticktext'] = [traitAssociation.length.toString().concat(" Variants ")]
      return new_fig;
  }
};

window.dash_clientside.clientside3 = {
    update_figure3: function(inputValues,store) {
      var x_arrow_val = inputValues[0];
      var x_arrow_start = inputValues[1];
      var y_arrow_val = inputValues[2];
      var y_arrow_start = inputValues[3];

      var fig = store;
      var new_fig = {};

      new_fig['layout'] = fig['layout'];
      new_fig['layout']['annotations'][0]['x'] = x_arrow_val;
      new_fig['layout']['annotations'][0]['ax'] = x_arrow_start;
      new_fig['layout']['annotations'][0]['y'] = y_arrow_val;
      new_fig['layout']['annotations'][0]['ay'] = y_arrow_start;
      return new_fig;
  }
};
  //
	// var fig = store[0];
	// if (!rows) {
  //          throw "Figure data not loaded, aborting update."
  //      }
	// var new_fig = {};
	// new_fig['data'] = [];
	// new_fig['layout'] = fig['layout'];
	// var countries = [];
	// var max = 100;
	// var max_data = 0;
	// for (i = 0; i < selectedrows.length; i++) {
	//     countries.push(rows[selectedrows[i]]["country_region"]);
	// }
	// if (cases_type === 'active'){
	//     new_fig['layout']['annotations'][0]['visible'] = false;
	//     new_fig['layout']['annotations'][1]['visible'] = true;
	//     for (i = 0; i < fig['data'].length; i++) {
	// 	var name = fig['data'][i]['name'];
	// 	if (countries.includes(name) || countries.includes(name.substring(1))){
	// 	    new_fig['data'].push(fig['data'][i]);
	// 	    max_data = Math.max(...fig['data'][i]['y']);
	// 	    if (max_data > max){
	// 		max = max_data;
	// 	    }
	// 	}
	//     }
	// }
	// else{
	//     new_fig['layout']['annotations'][0]['visible'] = true;
	//     new_fig['layout']['annotations'][1]['visible'] = false;
	//     for (i = 0; i < fig['data'].length; i++) {
	// 	var name = fig['data'][i]['name'];
	// 	if (countries.includes(name.substring(2))){
	// 	    new_fig['data'].push(fig['data'][i]);
	// 	    max_data = Math.max(...fig['data'][i]['y']);
	// 	    if (max_data > max){
	// 		max = max_data;
	// 	    }
	// 	}
	//     }
	// }
	// new_fig['layout']['yaxis']['type'] = log_or_lin;
	// if (log_or_lin === 'log'){
	//     new_fig['layout']['legend']['x'] = .65;
	//     new_fig['layout']['legend']['y'] = .1;
	//     new_fig['layout']['yaxis']['range'] = [1.2, Math.log10(max)];
	//     new_fig['layout']['yaxis']['autorange'] = false;
	// }
	// else{
	//     new_fig['layout']['legend']['x'] = .05;
	//     new_fig['layout']['legend']['y'] = .8;
	//     new_fig['layout']['yaxis']['autorange'] = true;
	// }
  //       return new_fig;
