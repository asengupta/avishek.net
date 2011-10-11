<?php 
$root = dirname(dirname(dirname(dirname(dirname(dirname(dirname(__FILE__)))))));
if (file_exists($root.'/wp-load.php')) {
	// WP 2.6
	require_once($root.'/wp-load.php');
} else {
	echo 'alert("Cannot include '.$root.'/wp-load.php!");';
}
Header("content-type: application/x-javascript"); ?>

mySettings = {
	markupSet: [
		{name:'<?php _e('Bold', 'markitup'); ?>', key:'B', openWith:'[b]', closeWith:'[/b]'},
		{name:'<?php _e('Italic', 'markitup'); ?>', key:'I', openWith:'[i]', closeWith:'[/i]'},
		{name:'<?php _e('Underline', 'markitup'); ?>', key:'U', openWith:'[u]', closeWith:'[/u]'},
		{separator:'---------------' },
		{name:'<?php _e('Image', 'markitup'); ?>', key:'P', replaceWith:'[img][![<?php _e('URL', 'markitup'); ?>]!][/img]'},
		{name:'<?php _e('Link', 'markitup'); ?>', key:'L', openWith:'[url=[![<?php _e('URL', 'markitup'); ?>]!]]', closeWith:'[/url]', placeHolder:'<?php _e('Your text to link here...', 'markitup'); ?>'},
		{separator:'---------------' },
		{name:'<?php _e('Size', 'markitup'); ?>', key:'S', openWith:'[size=[![<?php _e('Text size', 'markitup'); ?>]!]]', closeWith:'[/size]',
		dropMenu :[
			{name:'<?php _e('Big', 'markitup'); ?>', openWith:'[size=200]', closeWith:'[/size]' },
			{name:'<?php _e('Normal', 'markitup'); ?>', openWith:'[size=100]', closeWith:'[/size]' },
			{name:'<?php _e('Small', 'markitup'); ?>', openWith:'[size=50]', closeWith:'[/size]' }
		]},
		{separator:'---------------' },
		{name:'<?php _e('Bulleted list', 'markitup'); ?>', openWith:'[list]\n', closeWith:'\n[/list]'},
		{name:'<?php _e('Numeric list', 'markitup'); ?>', openWith:'[list=[![<?php _e('Starting number', 'markitup'); ?>]!]]\n', closeWith:'\n[/list]'}, 
		{name:'<?php _e('List item', 'markitup'); ?>', openWith:'[*] '},
		{separator:'---------------' },
		{name:'<?php _e('Quote block', 'markitup'); ?>', openWith:'[quote]', closeWith:'[/quote]'},
		{name:'<?php _e('Code block', 'markitup'); ?>', openWith:'[code]', closeWith:'[/code]'}, 
		{separator:'---------------' },
		{name:'<?php _e('Wordpress More... tag', 'markitup'); ?>', openWith:'\n<!--more-->\n'},
		{separator:'---------------' },
		{name:'<?php _e('Clean', 'markitup'); ?>', className:"clean", replaceWith:function(markitup) { return markitup.selection.replace(/\[(.*?)\]/g, "") } }
	]
}