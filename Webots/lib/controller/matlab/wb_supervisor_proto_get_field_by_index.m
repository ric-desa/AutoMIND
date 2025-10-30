function result = wb_supervisor_proto_get_field_by_index(protoref, fieldindex)
% Usage: wb_supervisor_proto_get_field_by_index(protoref, fieldindex)
% Matlab API for Webots
% Online documentation is available <a href="https://www.cyberbotics.com/doc/reference/supervisor">here</a>

result = calllib('libController', 'wb_supervisor_proto_get_field_by_index', protoref, fieldindex);
