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
    nameSpace:           "textile", // Useful to prevent multi-instances CSS conflict
    previewParserPath:   "~/sets/textile/preview.php",
    onShiftEnter:        {keepDefault:false, replaceWith:'\n\n'},
    markupSet: [
        {name:'<?php _e('First level heading', 'markitup'); ?>', key:'1', openWith:'h1(!(([![<?php _e('Class', 'markitup'); ?>]!]))!). ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Second level heading', 'markitup'); ?>', key:'2', openWith:'h2(!(([![<?php _e('Class', 'markitup'); ?>]!]))!). ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading 3', 'markitup'); ?>', key:'3', openWith:'h3(!(([![<?php _e('Class', 'markitup'); ?>]!]))!). ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading ', 'markitup'); ?>', key:'4', openWith:'h4(!(([![<?php _e('Class', 'markitup'); ?>]!]))!). ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading 5', 'markitup'); ?>', key:'5', openWith:'h5(!(([![<?php _e('Class', 'markitup'); ?>]!]))!). ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Heading 6', 'markitup'); ?>', key:'6', openWith:'h6(!(([![<?php _e('Class', 'markitup'); ?>]!]))!). ', placeHolder:'<?php _e('Your title here...', 'markitup'); ?>' },
        {name:'<?php _e('Paragraph', 'markitup'); ?>', key:'P', openWith:'p(!(([![<?php _e('Class', 'markitup'); ?>]!]))!). '}, 
        {separator:'---------------' },
        {name:'<?php _e('Bold', 'markitup'); ?>', key:'B', closeWith:'*', openWith:'*'}, 
        {name:'<?php _e('Italic', 'markitup'); ?>', key:'I', closeWith:'_', openWith:'_'}, 
        {name:'<?php _e('Stroke through', 'markitup'); ?>', key:'S', closeWith:'-', openWith:'-'}, 
        {separator:'---------------' },
        {name:'<?php _e('Bulleted list', 'markitup'); ?>', openWith:'(!(* |!|*)!)'}, 
        {name:'<?php _e('Numeric list', 'markitup'); ?>', openWith:'(!(# |!|#)!)'}, 
        {separator:'---------------' },
        {name:'<?php _e('Image', 'markitup'); ?>', replaceWith:'![![<?php _e('URL', 'markitup'); ?>:!:http://]!]([![<?php _e('Alternative text', 'markitup'); ?>]!])!'}, 
        {name:'<?php _e('Link', 'markitup'); ?>', openWith:'"', closeWith:'([![<?php _e('Title', 'markitup'); ?>]!])":[![<?php _e('Link', 'markitup'); ?>:!:http://]!]', placeHolder:'<?php _e('Your text to link here...', 'markitup'); ?>' },
        {separator:'---------------' },
        {name:'<?php _e('Quote block', 'markitup'); ?>', openWith:'bq(!(([![<?php _e('Class', 'markitup'); ?>]!])!)). '}, 
        {name:'<?php _e('Code block', 'markitup'); ?>', openWith:'@', closeWith:'@'},
		{separator:'---------------' },
		{name:'<?php _e('Wordpress More... tag', 'markitup'); ?>', openWith:'\n<!--more-->\n'}
    ]
}