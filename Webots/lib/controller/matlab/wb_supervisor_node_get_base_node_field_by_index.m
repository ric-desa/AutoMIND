function result = wb_supervisor_node_get_base_node_field_by_index(noderef, fieldindex)
% Usage: wb_supervisor_node_get_base_node_field_by_index(noderef, fieldindex)
% Matlab API for Webots
% Online documentation is available <a href="https://www.cyberbotics.com/doc/reference/supervisor">here</a>

result = calllib('libController', 'wb_supervisor_node_get_base_node_field_by_index', noderef, fieldindex);
