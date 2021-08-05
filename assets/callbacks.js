if (!window.dash_clientside) {
    window.dash_clientside = {}
}
window.dash_clientside.clientsideAF = {
    update_affigure: function(indexes, store) {
      var sim_idxs = indexes[0];
      var colors = indexes[1];
      var fig = store;
      var new_fig = {};

        new_fig['layout'] = fig['layout'];
        new_fig['data'] = [];
        sim_idxs.forEach(function(item, index) { //for (i = 0; i < 3; i++) { // fig['data'].length
		        new_fig['data'].push(fig['data'][item]);
            new_fig['data'][index]['line']['color']= colors[index]
        });
      return new_fig;
    }
};

window.dash_clientside.clientsideDist = {
    update_distfigure: function(inputValues,store) {
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

window.dash_clientside.clientsideArrow = {
    update_Arrowfigure: function(inputValues,store) {
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
