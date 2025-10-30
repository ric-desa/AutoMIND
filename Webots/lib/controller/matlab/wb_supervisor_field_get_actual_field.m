function result = wb_supervisor_field_get_actual_field(fieldref)
% Usage: wb_supervisor_field_get_actual_field(fieldref)
% Matlab API for Webots
% Online documentation is available <a href="https://www.cyberbotics.com/doc/reference/supervisor">here</a>

result = calllib('libController', 'wb_supervisor_field_get_actual_field', fieldref);
