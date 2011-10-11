<?php
/*
Plugin Name: WP MarkItUp!
Plugin URI: http://wpmarkitup.stefanoverna.com/
Description: Replaces the Wordpress textarea + "quicktags" toolbar with <a href="http://markitup.jaysalvat.com/">MarkItUp!</a>, the jQuery lightweight, customizable and flexible markup editor by <a href="http://www.jaysalvat.com/" target="_blank">Jay Salvat</a>.
Version: 1.3.5
Author: Stefano Verna
Author URI: http://www.stefanoverna.com/
*/
/*
Copyright (C) 2008  Stefano Verna  (email : stefano.verna@gmail.com)

Based on the great "FCKEditor for Wordpress" plugin by Dean Lee (email : deanlee2@hotmail.com)

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

require_once('markitup_class.php');
add_action('admin_menu', array(&$markitup, 'add_option_page'));
add_action('admin_head', array(&$markitup, 'add_admin_head'));
add_action('edit_form_advanced', array(&$markitup, 'load_markitup'));
add_action('edit_page_form', array(&$markitup, 'load_markitup'));
add_action('simple_edit_form', array(&$markitup, 'load_markitup'));
register_activation_hook(basename(dirname(__FILE__)).'/' . basename(__FILE__), array(&$markitup, 'activate'));
register_deactivation_hook(basename(dirname(__FILE__)).'/' . basename(__FILE__), array(&$markitup, 'deactivate'));

?>
