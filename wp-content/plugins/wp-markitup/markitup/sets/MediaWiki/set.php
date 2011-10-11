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
	onShiftEnter:		{keepDefault:false, replaceWith:'\n\n'},
	markupSet: [
		{name:'<?php _e('First level heading', 'markitup'); ?>', key:'1', openWith:'== ', closeWith:' ==', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{name:'<?php _e('Second level heading', 'markitup'); ?>', key:'2', openWith:'=== ', closeWith:' ===', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{name:'<?php _e('Heading 3', 'markitup'); ?>', key:'3', openWith:'==== ', closeWith:' ====', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{name:'<?php _e('Heading 4', 'markitup'); ?>', key:'4', openWith:'===== ', closeWith:' =====', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{name:'<?php _e('Heading 5', 'markitup'); ?>', key:'5', openWith:'====== ', closeWith:' ======', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
		{separator:'---------------' },		
		{name:'<?php _e('Bold', 'markitup'); ?>', key:'B', openWith:"'''", closeWith:"'''"}, 
		{name:'<?php _e('Italic', 'markitup'); ?>', key:'I', openWith:"''", closeWith:"''"}, 
		{name:'<?php _e('Stroke through', 'markitup'); ?>', key:'S', openWith:'<s>', closeWith:'</s>'}, 
		{separator:'---------------' },
		{name:'<?php _e('Bulleted list', 'markitup'); ?>', openWith:'(!(* |!|*)!)'}, 
		{name:'<?php _e('Numeric list', 'markitup'); ?>', openWith:'(!(# |!|#)!)'}, 
		{separator:'---------------' },
		{name:'<?php _e('Image', 'markitup'); ?>', key:"P", replaceWith:'[[Image:[![<?php _e('URL', 'markitup'); ?>:!:http://]!]|[![<?php _e('Alternative text', 'markitup'); ?>]!]]]'}, 
		{name:'<?php _e('Link', 'markitup'); ?>', key:"L", openWith:"[[![<?php _e('URL', 'markitup'); ?>]!] ", closeWith:']', placeHolder:'<?php _e('Your text to link here...', 'markitup'); ?>' },
		{separator:'---------------' },
		{name:'<?php _e('Quote block', 'markitup'); ?>', openWith:'(!(> |!|>)!)', placeHolder:''},
		{name:'<?php _e('Code block', 'markitup'); ?>', openWith:'(!(<source lang="[![<?php _e('Language', 'markitup'); ?>:!:php]!]">|!|<pre>)!)', closeWith:'(!(</source>|!|</pre>)!)'},
		{separator:'---------------' },
		{name:'<?php _e('Wordpress More... tag', 'markitup'); ?>', openWith:'\n<!--more-->\n'}
	]
}